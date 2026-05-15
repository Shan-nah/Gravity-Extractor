const express = require('express');
const puppeteer = require('puppeteer');
const axios = require('axios');
const pdfParse = require('pdf-parse');
const mammoth = require('mammoth');
const multer = require('multer');
const PDFDocument = require('pdfkit');
const cors = require('cors');
const fs = require('fs');
const fsp = fs.promises;
const path = require('path');
const os = require('os');
const { GoogleGenerativeAI } = require("@google/generative-ai");


const app = express();
const PORT = process.env.PORT || 5002;

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname, 'public')));
app.use('/output', express.static(path.join(__dirname, 'output')));

const outputDir = path.join(__dirname, 'output');
if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

// ─── SSE progress helper ──────────────────────────────────────────────────────
function sseWrite(res, event, data) {
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

// ─── Key loading ──────────────────────────────────────────────────────────────
// Keys are read from (in order of priority):
//   1. GEMINI_API_KEY environment variable (set this in Railway/Render dashboard)
//   2. keys.txt file (local development)

let GEMINI_KEYS = [];

if (process.env.GEMINI_API_KEY) {
  const envKeys = process.env.GEMINI_API_KEY.split(',').map(k => k.trim()).filter(k => k.startsWith('AIza'));
  GEMINI_KEYS = [...new Set(envKeys)];
  console.log(`Loaded ${GEMINI_KEYS.length} Gemini key(s) from environment.`);
}

const keysFilePath = path.join(__dirname, 'keys.txt');
if (fs.existsSync(keysFilePath)) {
  try {
    const lines = fs.readFileSync(keysFilePath, 'utf8').split('\n').map(k => k.trim());
    const geminiLines = lines.filter(k => k.startsWith('AIza'));
    if (geminiLines.length > 0) {
      GEMINI_KEYS = [...new Set([...GEMINI_KEYS, ...geminiLines])];
      console.log(`Loaded ${geminiLines.length} Gemini key(s) from keys.txt.`);
    }
  } catch (err) {
    console.error('Failed to read keys.txt:', err.message);
  }
}

// ─── Routes ───────────────────────────────────────────────────────────────────

app.get('/api/extract', async (req, res) => {
  const { url, apiKey } = req.query;
  const geminiKey = (apiKey && apiKey.trim()) ? apiKey.trim() : GEMINI_KEYS[0];

  if (!url || !geminiKey) {
    const msg = !url
      ? 'url parameter is required'
      : 'No Gemini API key available. Paste one on the page or add it to keys.txt.';
    res.status(400).json({ error: msg });
    return;
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  const tempFiles = [];

  try {
    // 1 ─ Scrape
    sseWrite(res, 'progress', { step: 1, message: 'Opening tender page and finding documents…' });
    const { pageText, documentLinks, pageTitle } = await scrapeTenderPage(url);
    sseWrite(res, 'progress', { step: 1, message: `Found ${documentLinks.length} document link(s) on the page.` });

    // 2 ─ Download (all documents, no cap)
    sseWrite(res, 'progress', { step: 2, message: `Downloading ${documentLinks.length} document(s)…` });
    const downloaded = await downloadDocuments(documentLinks);
    tempFiles.push(...downloaded.map(f => f.path));
    sseWrite(res, 'progress', { step: 2, message: `Downloaded ${downloaded.length} document(s) successfully.` });

    // 3 ─ Extract text
    sseWrite(res, 'progress', { step: 3, message: 'Extracting text from documents…' });
    const docTexts = await extractTextFromDocuments(downloaded);
    sseWrite(res, 'progress', { step: 3, message: `Extracted text from ${docTexts.length} document(s).` });

    // 4 ─ Gemini analysis
    sseWrite(res, 'progress', { step: 4, message: 'Analysing with Gemini — reading all documents in full…' });
    const report = await processWithGemini(geminiKey, pageText, docTexts, url, pageTitle);
    sseWrite(res, 'progress', { step: 4, message: 'AI analysis complete.' });

    // 5 ─ PDF
    sseWrite(res, 'progress', { step: 5, message: 'Generating PDF report…' });
    const pdfFilename = `tender_${Date.now()}.pdf`;
    const pdfPath = path.join(outputDir, pdfFilename);
    await generatePDF(report, url, pageTitle, pdfPath);
    sseWrite(res, 'progress', { step: 5, message: 'PDF ready!' });

    sseWrite(res, 'done', { pdfUrl: `/output/${pdfFilename}`, docsFound: downloaded.length });

  } catch (err) {
    console.error('Error:', err);
    sseWrite(res, 'error', { message: err.message || 'Unknown error' });
  } finally {
    for (const f of tempFiles) {
      try { await fsp.unlink(f); } catch (_) {}
    }
    res.end();
  }
});

// ─── File upload endpoints ────────────────────────────────────────────────────

const upload = multer({
  dest: os.tmpdir(),
  limits: { fileSize: 50 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    const allowed = /\.(pdf|doc|docx|xls|xlsx|zip)$/i;
    cb(null, allowed.test(path.extname(file.originalname)));
  },
});

// Temporary store: jobId → array of file descriptors
const uploadJobs = new Map();

// Step 1: receive the files, return a job ID immediately
app.post('/api/upload', upload.array('files', 20), (req, res) => {
  if (!req.files || req.files.length === 0) {
    return res.status(400).json({ error: 'No supported files uploaded (PDF, DOC, DOCX, XLS, XLSX, ZIP).' });
  }
  const jobId = `job_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  uploadJobs.set(jobId, req.files.map(f => ({
    path: f.path,
    ext: path.extname(f.originalname).toLowerCase() || '.pdf',
    name: f.originalname,
  })));
  // Auto-expire after 15 minutes in case the SSE step is never called
  setTimeout(() => {
    const leftover = uploadJobs.get(jobId);
    if (leftover) {
      leftover.forEach(f => fsp.unlink(f.path).catch(() => {}));
      uploadJobs.delete(jobId);
    }
  }, 15 * 60 * 1000);
  res.json({ jobId });
});

// Step 2: SSE stream — process the uploaded files and generate a PDF
app.get('/api/extract-upload', async (req, res) => {
  const { id, apiKey } = req.query;
  const files = uploadJobs.get(id);
  const geminiKey = (apiKey && apiKey.trim()) ? apiKey.trim() : GEMINI_KEYS[0];

  if (!files) return res.status(400).json({ error: 'Invalid or expired upload ID.' });
  if (!geminiKey) return res.status(400).json({ error: 'No Gemini API key available. Paste one on the page or add it to keys.txt.' });

  uploadJobs.delete(id); // claim it — process once only

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  try {
    sseWrite(res, 'progress', { step: 1, message: `${files.length} file(s) received — ready to process.` });
    sseWrite(res, 'progress', { step: 2, message: 'Files staged for extraction.' });

    sseWrite(res, 'progress', { step: 3, message: 'Extracting text from uploaded file(s)…' });
    const docTexts = await extractTextFromDocuments(files);
    sseWrite(res, 'progress', { step: 3, message: `Text extracted from ${docTexts.length} document(s).` });

    if (docTexts.length === 0) throw new Error('Could not extract any readable text from the uploaded file(s).');

    sseWrite(res, 'progress', { step: 4, message: 'Analysing with Gemini…' });
    const report = await processWithGemini(geminiKey, '', docTexts, 'uploaded-document', files[0].name);
    sseWrite(res, 'progress', { step: 4, message: 'AI analysis complete.' });

    sseWrite(res, 'progress', { step: 5, message: 'Generating PDF report…' });
    const pdfFilename = `tender_${Date.now()}.pdf`;
    const pdfPath = path.join(outputDir, pdfFilename);
    const sourceLabel = files.map(f => f.name).join(', ');
    await generatePDF(report, `Uploaded: ${sourceLabel}`, files[0].name, pdfPath);
    sseWrite(res, 'progress', { step: 5, message: 'PDF ready!' });

    sseWrite(res, 'done', { pdfUrl: `/output/${pdfFilename}`, docsFound: files.length });
  } catch (err) {
    console.error('Upload extraction error:', err);
    sseWrite(res, 'error', { message: err.message || 'Unknown error' });
  } finally {
    files.forEach(f => fsp.unlink(f.path).catch(() => {}));
    res.end();
  }
});

// ─── Scraper ──────────────────────────────────────────────────────────────────

async function scrapeTenderPage(url) {
  const browser = await puppeteer.launch({
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-gpu',
      '--disable-blink-features=AutomationControlled',
    ],
  });

  try {
    const page = await browser.newPage();

    await page.setUserAgent(
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    );
    await page.setExtraHTTPHeaders({
      'Accept-Language': 'en-US,en;q=0.9',
      Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    });

    await page.setRequestInterception(true);
    page.on('request', (req) => {
      const type = req.resourceType();
      if (['image', 'media', 'font', 'stylesheet'].includes(type)) req.abort();
      else req.continue();
    });

    await page.goto(url, { waitUntil: 'networkidle2', timeout: 90000 });
    await new Promise(r => setTimeout(r, 4000));

    const pageTitle = await page.title();
    const pageText = await page.evaluate(() => document.body.innerText || document.body.textContent || '');

    const documentLinks = await page.evaluate(() => {
      const seen = new Set();
      const results = [];

      document.querySelectorAll('a[href]').forEach((a) => {
        const href = a.href;
        const text = (a.textContent || '').trim();
        if (!href || seen.has(href)) return;

        const isDoc =
          /\.(pdf|doc|docx|xls|xlsx|zip|rar|7z)(\?.*)?$/i.test(href) ||
          /\/(download|document|attachment|file|bid|tender.?doc)/i.test(href) ||
          /download/i.test(text) ||
          /tender.?doc/i.test(text) ||
          /corrigendum/i.test(text) ||
          /notice.?inviting/i.test(text);

        if (isDoc) {
          seen.add(href);
          results.push({ url: href, text });
        }
      });

      return results;
    });

    return { pageText, documentLinks, pageTitle };
  } finally {
    await browser.close();
  }
}

// ─── Downloader ───────────────────────────────────────────────────────────────

async function downloadDocuments(links) {
  const tempDir = path.join(os.tmpdir(), `gravity_${Date.now()}`);
  fs.mkdirSync(tempDir, { recursive: true });

  const downloaded = [];

  for (const link of links) {
    try {
      const response = await axios.get(link.url, {
        responseType: 'arraybuffer',
        timeout: 45000,
        maxContentLength: 50 * 1024 * 1024,
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
          Accept: '*/*',
        },
      });

      const ct = (response.headers['content-type'] || '').toLowerCase();
      let ext = path.extname(new URL(link.url).pathname) || '.pdf';
      if (ct.includes('pdf')) ext = '.pdf';
      else if (ct.includes('officedocument.wordprocessingml')) ext = '.docx';
      else if (ct.includes('msword')) ext = '.doc';
      else if (ct.includes('zip')) ext = '.zip';

      const fname = `doc_${Date.now()}_${Math.random().toString(36).slice(2, 7)}${ext}`;
      const fpath = path.join(tempDir, fname);
      await fsp.writeFile(fpath, Buffer.from(response.data));
      downloaded.push({ path: fpath, ext, name: link.text || fname });
      console.log(`  ✓ downloaded: ${link.url.split('/').pop()}`);
    } catch (err) {
      console.warn(`  ✗ skip ${link.url}: ${err.message}`);
    }
  }

  return downloaded;
}

// ─── Text extractor ───────────────────────────────────────────────────────────

async function extractTextFromDocuments(files) {
  const results = [];
  for (const file of files) {
    try {
      let text = '';
      if (file.ext === '.pdf') {
        const buf = await fsp.readFile(file.path);
        const data = await pdfParse(buf);
        text = data.text;
      } else if (['.doc', '.docx'].includes(file.ext)) {
        const result = await mammoth.extractRawText({ path: file.path });
        text = result.value;
      }
      if (text.trim().length > 50) {
        results.push({ name: file.name, text });
      }
    } catch (err) {
      console.warn(`  text-extract fail (${file.path}): ${err.message}`);
    }
  }
  return results;
}

// ─── Gemini ───────────────────────────────────────────────────────────────────

const GEMINI_MODELS = [
  'gemini-2.5-flash',      // primary
  'gemini-2.5-flash-lite', // fallback
  'gemini-3.1-flash-lite', // fallback
  'gemma-4-26b-a4b-it',    // last resort
  'gemma-4-31b-it',        // last resort
];

const EXTRACTION_PROMPT = `You are a tender intelligence analyst. Read every document and produce a concise briefing for a contractor deciding whether to bid.

FORMATTING RULES — non-negotiable:
- Use ## markdown headings for ALL section names (e.g. ## Financial Figures). NEVER use **Bold:** as a section heading.
- If a section has no data, OMIT IT ENTIRELY — do not write the heading, do not write "Not specified" or "N/A".
- Use | Field | Value | tables for structured data (max 3 columns per table).
- Use numbered or bulleted lists for checklists and scope items. Each point = 1 line maximum.
- Zero prose paragraphs. Numbers and dates must be verbatim from the source.
- Each fact appears once only — no repetition across sections.
- Tables with many columns (e.g. manpower breakdowns) MUST be split into separate 2-column tables, one per attribute (e.g. one table for roles+quantities, another for qualifications, another for wages). Never create a table with more than 3 columns.

ALWAYS INCLUDE (only if data exists):
1. ## Core Details — title, reference no., authority, location (2-col table)
2. ## Key Dates — all deadlines with exact date + time (2-col table)
3. ## Financial Figures — all amounts with calculation basis (2-col table)
4. ## Eligibility Criteria — turnover, experience, personnel, certifications (2-col table)
5. ## Scope of Work — numbered bullets: what + quantity + where
6. ## Submission Documents — numbered list of every mandatory document
7. ## Contract Terms — LD, payment, validity, penalties (2-col table)
8. ## Executive Summary — 5 bullets max, most critical facts only

For anything unique to this tender (BOQ, manpower breakdown, technical specs, penalties), add it as its own ## section. Adapt to what the tender actually contains.`;

async function geminiCall(apiKey, modelName, prompt) {
  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({ model: modelName });
  const result = await model.generateContent([prompt]);
  return result.response.text();
}

async function processWithGemini(apiKey, pageText, docTexts, url, pageTitle) {
  const buildCombined = (page, docs) => [
    page ? `SOURCE: Web Page (${pageTitle || url})\nTEXT:\n${page}` : null,
    ...docs.map(d => `SOURCE: Document "${d.name}"\nTEXT:\n${d.text}`)
  ].filter(Boolean).join('\n\n---\n\n');

  const fullCombined = buildCombined(pageText, docTexts);
  const charCount = fullCombined.length;
  console.log(`  [Gemini] Total input: ${charCount.toLocaleString()} chars (~${Math.round(charCount / 4).toLocaleString()} tokens)`);

  // 500K chars ≈ 125K tokens — safely within Gemma's 262K limit and
  // well within Gemini's 1M token limit. Larger docs get split into chunks.
  const CHUNK_SIZE = 500_000;

  const rawExtractPrompt = `${EXTRACTION_PROMPT}\n\nThis is a partial extraction pass — extract every fact, figure, date, and number you find. Do not produce a final formatted report yet.\n\nRAW CONTENT:\n`;

  let lastError = null;

  for (const modelName of GEMINI_MODELS) {
    try {
      console.log(`  [Gemini] Attempting with model: ${modelName}…`);

      if (charCount <= CHUNK_SIZE) {
        // Single-pass — content fits in one call
        const prompt = `${EXTRACTION_PROMPT}\n\nRAW CONTENT:\n${fullCombined}`;
        const report = await geminiCall(apiKey, modelName, prompt);
        console.log(`  [Gemini] Extraction complete (${modelName}).`);
        return report;
      }

      // Multi-pass chunking — split content into CHUNK_SIZE pieces
      const chunks = [];
      for (let start = 0; start < fullCombined.length; start += CHUNK_SIZE) {
        chunks.push(fullCombined.slice(start, start + CHUNK_SIZE));
      }
      console.log(`  [Gemini] Content too large — splitting into ${chunks.length} chunk(s) of ~${Math.round(CHUNK_SIZE / 1000)}K chars each.`);

      const extractions = [];
      for (let i = 0; i < chunks.length; i++) {
        console.log(`  [Gemini] Pass ${i + 1}/${chunks.length}: extracting chunk…`);
        const extraction = await geminiCall(apiKey, modelName, rawExtractPrompt + chunks[i]);
        extractions.push(extraction);
      }

      // Merge all chunk extractions into a final report
      let merged = extractions[0];
      for (let i = 1; i < extractions.length; i++) {
        console.log(`  [Gemini] Merging chunk ${i + 1} into report…`);
        const mergePrompt = `${EXTRACTION_PROMPT}\n\nMerge the two partial extractions below into one final concise report. Prefer more specific or larger values when the same field appears in both. No data loss.\n\nPART 1:\n${merged}\n\nPART 2:\n${extractions[i]}`;
        merged = await geminiCall(apiKey, modelName, mergePrompt);
      }

      console.log(`  [Gemini] Multi-pass extraction complete (${modelName}, ${chunks.length} chunk(s)).`);
      return merged;

    } catch (err) {
      console.warn(`  [Gemini] Model ${modelName} failed: ${err.message}`);
      lastError = err;
    }
  }

  throw new Error(`All Gemini models failed. Last error: ${lastError?.message}`);
}

// ─── PDF helpers ──────────────────────────────────────────────────────────────

function extractCoverField(markdown, ...keys) {
  for (const key of keys) {
    const pattern = new RegExp(`\\|\\s*\\*{0,2}${key}\\*{0,2}\\s*\\|\\s*([^|\\n]+)`, 'i');
    const m = markdown.match(pattern);
    if (m) {
      const val = m[1].trim().replace(/\*\*/g, '');
      if (val && val !== '-' && val !== '' && !/^\s*$/.test(val)) return val;
    }
  }
  return null;
}


function h2AccentColor(text) {
  const t = text.toUpperCase();
  if (/OVERVIEW|IDENTIFICATION|PROJECT/.test(t)) return '#2b6cb0';
  if (/DATE|TIMELINE|DEADLINE/.test(t)) return '#276749';
  if (/FINANCIAL|FUND|AMOUNT|PAYMENT/.test(t)) return '#b7791f';
  if (/SCOPE|TECHNICAL|SPECIFICATION|BOQ|BILL|SCHEDULE/.test(t)) return '#553c9a';
  if (/DOCUMENT|CHECKLIST|SUBMISSION/.test(t)) return '#2c7a7b';
  if (/TERM|CONDITION|EVALUATION|CLAUSE|CRITERIA/.test(t)) return '#9b2c2c';
  if (/EXECUTIVE|STRATEGIC|INSIGHT|SUMMARY/.test(t)) return '#1a365d';
  if (/ELIGIB|QUALIF|BIDDER/.test(t)) return '#276749';
  if (/AUTHORITY|ISSUING/.test(t)) return '#2b6cb0';
  return '#2d3748';
}

// ─── PDF generator ────────────────────────────────────────────────────────────

async function generatePDF(markdownContent, sourceUrl, pageTitle, outputPath) {
  return new Promise((resolve, reject) => {
    const doc = new PDFDocument({
      margin: 52,
      size: 'A4',
      bufferPages: true,
      info: {
        Title: 'Tender Information Report',
        Author: 'Gravity Extractor',
        Subject: pageTitle || 'Tender Analysis',
      },
    });

    const stream = fs.createWriteStream(outputPath);
    doc.pipe(stream);

    const pageW = doc.page.width;
    const contentW = pageW - 104;

    // ── Cover page ──────────────────────────────────────────────────────────
    doc.rect(0, 0, pageW, 120).fill('#1a365d');
    doc.fillColor('#ffffff').fontSize(22).font('Helvetica-Bold')
       .text('TENDER INFORMATION REPORT', 52, 30, { align: 'center', width: contentW });
    doc.fontSize(10).font('Helvetica')
       .text('Gravity Extractor  ·  Powered by Gemini AI', 52, 64, { align: 'center', width: contentW });
    doc.fontSize(9)
       .text(new Date().toLocaleString('en-IN'), 52, 82, { align: 'center', width: contentW });

    // Info box — pull key fields from extracted markdown
    const tenderTitle = extractCoverField(markdownContent, 'Tender Title', 'Project Name', 'Title');
    const authority = extractCoverField(markdownContent, 'Organization / Department', 'Organization', 'Authority', 'Department');
    const refNo = extractCoverField(markdownContent, 'Tender Reference / NIT No\\.', 'Reference No\\.', 'NIT No\\.', 'Tender Reference');
    const deadline = extractCoverField(markdownContent, 'Bid Submission Deadline', 'Submission Deadline', 'Last Date');

    const infoBoxY = 136;
    const infoItems = [
      tenderTitle   ? { label: 'Tender', value: tenderTitle }   : null,
      refNo         ? { label: 'Reference No.', value: refNo }   : null,
      authority     ? { label: 'Authority', value: authority }   : null,
      deadline      ? { label: 'Submission Deadline', value: deadline } : null,
    ].filter(Boolean);

    if (infoItems.length > 0) {
      const boxH = infoItems.length * 26 + 16;
      doc.rect(52, infoBoxY, contentW, boxH).fill('#ebf4ff');
      doc.rect(52, infoBoxY, 4, boxH).fill('#2b6cb0');

      infoItems.forEach((item, idx) => {
        const iy = infoBoxY + 10 + idx * 26;
        doc.fillColor('#1a365d').fontSize(9).font('Helvetica-Bold')
           .text(`${item.label}: `, 64, iy, { continued: true, width: contentW - 16 });
        doc.fillColor('#2d3748').font('Helvetica')
           .text(item.value, { continued: false });
      });

      doc.y = infoBoxY + boxH + 12;
    } else {
      doc.y = infoBoxY + 12;
    }

    doc.fillColor('#888').fontSize(8).font('Helvetica')
       .text(`Source: ${sourceUrl}`, 52, doc.y, { width: contentW, ellipsis: true });
    doc.moveDown(0.8);
    doc.moveTo(52, doc.y).lineTo(543, doc.y).lineWidth(2).stroke('#1a365d');
    doc.moveDown(1);

    // ── Report content flows directly after the cover header — no forced page breaks ──
    renderMarkdown(doc, markdownContent.trim());

    // ── Page numbers ────────────────────────────────────────────────────────
    // Root cause of ghost pages: doc.text() at Y > (page.height - margin) triggers
    // PDFKit to auto-create a new blank page. Fix: temporarily open the bottom margin
    // so the footer Y is within bounds during text rendering.
    const { start, count } = doc.bufferedPageRange();

    // Detect trailing empty pages (y still at top margin = nothing was written there)
    let pageCount = count;
    while (pageCount > 1) {
      doc.switchToPage(start + pageCount - 1);
      if (doc.y <= doc.page.margins.top + 8) pageCount--;
      else break;
    }

    for (let i = 0; i < pageCount; i++) {
      doc.switchToPage(start + i);
      const savedBottom = doc.page.margins.bottom;
      doc.page.margins.bottom = 8; // temporarily expand content area to include footer zone
      const footerY = doc.page.height - 24;
      doc.fillColor('#aaaaaa').fontSize(8).font('Helvetica')
         .text(`Gravity Extractor  ·  Page ${i + 1} of ${pageCount}`, 52, footerY, {
           align: 'center',
           width: contentW,
           lineBreak: false,
         });
      doc.page.margins.bottom = savedBottom; // restore
    }

    doc.end();
    stream.on('finish', () => resolve(outputPath));
    stream.on('error', reject);
  });
}

function renderMarkdown(doc, content) {
  const lines = content.split('\n');
  let i = 0;

  while (i < lines.length) {
    const rawLine = lines[i];
    const trimmed = rawLine.trim();

    if (!trimmed) {
      doc.moveDown(0.3);
      i++;
      continue;
    }

    // ─── Table Handler ───
    if (trimmed.startsWith('|')) {
      const tableRows = [];
      while (i < lines.length && lines[i].trim().startsWith('|')) {
        const row = lines[i].trim()
          .split('|')
          .filter((_, idx, arr) => idx > 0 && idx < arr.length - 1)
          .map(cell => cell.trim());

        if (row.length > 0 && !row.every(cell => /^[:\s-]*$/.test(cell))) {
          tableRows.push(row);
        }
        i++;
      }

      if (tableRows.length > 0) {
        doc.moveDown(0.5);
        const colCount = tableRows[0].length;
        const minSafeColWidth = 55; // px — below this, table grid breaks badly
        const effectiveColWidth = 491 / colCount;

        if (colCount > 3 || effectiveColWidth < minSafeColWidth) {
          // ── Wide table fallback: render as stacked key-value pairs ──────────
          // Each data row becomes a group of "Header: Value" lines,
          // separated by a thin rule. This is always readable regardless of columns.
          const headers = tableRows[0].map(h => h.replace(/\*\*/g, '').trim());

          for (let rowIdx = 1; rowIdx < tableRows.length; rowIdx++) {
            const row = tableRows[rowIdx];
            if (rowIdx > 1) {
              // thin rule between records
              doc.moveTo(52, doc.y).lineTo(543, doc.y).lineWidth(0.3).stroke('#e2e8f0');
              doc.moveDown(0.2);
            }
            row.forEach((cell, ci) => {
              const key = headers[ci] || '';
              const val = cell.replace(/\*\*/g, '').trim();
              if (!val || val === '-') return;
              doc.fillColor('#1a4a8a').fontSize(9).font('Helvetica-Bold')
                 .text(`${key}: `, 56, doc.y, { continued: true, width: 487 });
              doc.fillColor('#222222').font('Helvetica').text(val, { lineGap: 1 });
            });
            doc.moveDown(0.25);
          }
          doc.moveDown(0.5);
        } else {
          // ── Normal table grid ──────────────────────────────────────────────
          // 2-col: 38/62 split; 3-col: equal thirds
          const colWidths = colCount === 2
            ? [491 * 0.38, 491 * 0.62]
            : Array(colCount).fill(491 / colCount);

          let dataRowIndex = 0;

          for (let rowIndex = 0; rowIndex < tableRows.length; rowIndex++) {
            const row = tableRows[rowIndex];
            const isHeader = rowIndex === 0;

            let maxCellHeight = 0;
            row.forEach((cell, ci) => {
              const cw = colWidths[ci] || (491 / colCount);
              const h = doc.heightOfString(cell.replace(/\*\*/g, ''), { width: cw - 16 });
              if (h > maxCellHeight) maxCellHeight = h;
            });
            const rowHeight = maxCellHeight + (isHeader ? 16 : 12);

            if (doc.y + rowHeight > doc.page.height - 52) doc.addPage();

            const currentY = doc.y;

            if (isHeader) {
              doc.rect(52, currentY, 491, rowHeight).fill('#edf2f7');
            } else {
              doc.rect(52, currentY, 491, rowHeight).fill(dataRowIndex % 2 === 1 ? '#f7fafc' : '#ffffff');
              dataRowIndex++;
            }

            let xCursor = 52;
            row.forEach((cell, colIndex) => {
              const cw = colWidths[colIndex] || (491 / colCount);
              doc.fillColor(isHeader ? '#1a365d' : '#222222')
                 .fontSize(9)
                 .font(isHeader ? 'Helvetica-Bold' : 'Helvetica')
                 .text(cell.replace(/\*\*/g, ''), xCursor + 8, currentY + 6, {
                   width: cw - 16,
                   align: 'left',
                 });
              doc.moveTo(xCursor, currentY).lineTo(xCursor, currentY + rowHeight).lineWidth(0.5).stroke('#e2e8f0');
              xCursor += cw;
            });

            doc.moveTo(52 + 491, currentY).lineTo(52 + 491, currentY + rowHeight).lineWidth(0.5).stroke('#e2e8f0');
            doc.moveTo(52, currentY).lineTo(52 + 491, currentY).lineWidth(0.5).stroke('#e2e8f0');
            doc.moveTo(52, currentY + rowHeight).lineTo(52 + 491, currentY + rowHeight).lineWidth(0.5).stroke('#e2e8f0');
            if (isHeader) {
              doc.moveTo(52, currentY + rowHeight).lineTo(52 + 491, currentY + rowHeight).lineWidth(2).stroke('#90cdf4');
            }

            doc.y = currentY + rowHeight;
          }
          doc.moveDown(0.5);
        }

        // CRITICAL: reset x to left margin after any table so subsequent
        // content doesn't render in a narrow right-side strip
        doc.x = doc.page.margins.left;
      }
      continue;
    }

    // ─── H1 ───
    if (trimmed.startsWith('# ') && !trimmed.startsWith('## ')) {
      const text = trimmed.slice(2).replace(/\*\*/g, '');
      doc.moveDown(1);
      if (doc.y + 40 > doc.page.height - doc.page.margins.bottom) doc.addPage();
      const y = doc.y;
      doc.rect(52, y, 491, 26).fill('#1a365d');
      doc.fillColor('#ffffff').fontSize(13).font('Helvetica-Bold')
         .text(text, 60, y + 7, { width: 475, lineBreak: false });
      doc.moveDown(1.4);
    }
    // ─── H2 ───
    else if (trimmed.startsWith('## ') && !trimmed.startsWith('### ')) {
      const text = trimmed.slice(3).replace(/\*\*/g, '');
      const accentColor = h2AccentColor(text);
      doc.moveDown(0.9);
      if (doc.y + 50 > doc.page.height - doc.page.margins.bottom) doc.addPage();
      const y = doc.y;
      // Left accent bar
      doc.rect(52, y, 5, 20).fill(accentColor);
      doc.fillColor('#1a365d').fontSize(12).font('Helvetica-Bold')
         .text(text, 62, y + 4, { width: 481 });
      doc.moveDown(0.2);
      doc.moveTo(62, doc.y + 2).lineTo(543, doc.y + 2).lineWidth(0.8).stroke('#a0aec0');
      doc.moveDown(0.5);
    }
    // ─── H3 ───
    else if (trimmed.startsWith('### ')) {
      const text = trimmed.slice(4).replace(/\*\*/g, '');
      doc.moveDown(0.5);
      doc.fillColor('#2d3748').fontSize(11).font('Helvetica-Bold').text(text);
      doc.moveDown(0.2);
    }
    // ─── Bullet ───
    else if (/^[-*]\s+/.test(trimmed)) {
      const text = trimmed.replace(/^[-*]\s+/, '').replace(/\*\*/g, '').replace(/\*/g, '');
      doc.fillColor('#222222').fontSize(10).font('Helvetica')
         .text(`•  ${text}`, { indent: 18, lineGap: 2 });
    }
    // ─── Bold key: value ───
    else if (/^\*\*(.+?)\*\*:?\s*(.*)/.test(trimmed)) {
      const kvMatch = trimmed.match(/^\*\*(.+?)\*\*:?\s*(.*)/);
      const key = kvMatch[1].trim();
      const val = kvMatch[2].trim() || 'Not specified';
      doc.fillColor('#1a365d').fontSize(10).font('Helvetica-Bold')
         .text(`${key}: `, { continued: true });
      doc.fillColor('#222222').font('Helvetica').text(val, { continued: false, lineGap: 3 });
    }
    // ─── Plain text ───
    else {
      doc.fillColor('#222222').fontSize(10).font('Helvetica')
         .text(trimmed.replace(/\*\*/g, '').replace(/\*/g, ''), { lineGap: 2 });
    }

    i++;
  }
}

// ─── Start ────────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`\n  Gravity Extractor`);
  console.log(`  ➜  http://localhost:${PORT}`);
  console.log(`  Press Ctrl+C to stop\n`);
});
