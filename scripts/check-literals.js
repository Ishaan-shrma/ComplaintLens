import fs from "node:fs";
import path from "node:path";
import parser from "@babel/parser";
import _traverse from "@babel/traverse";

const traverse = _traverse.default || _traverse;

const ROOT = process.cwd();
const SRC_DIR = path.resolve(ROOT, "src");
const ALLOWLIST_PATH = path.resolve(ROOT, "scripts", "literal-allowlist.json");

let allowlist = { numbers: {}, strings: {} };
if (fs.existsSync(ALLOWLIST_PATH)) {
  allowlist = JSON.parse(fs.readFileSync(ALLOWLIST_PATH, "utf-8"));
}

const CHECKED_PROPS = new Set(["value", "label", "delta", "data", "title", "caption"]);
const METRIC_STRING_PATTERN = /(\b\d+(\.\d+)?%|\b0\.\d{3,4}\b|\b\d{2,3},\d{3}\b)/;

const EXEMPT_FILES = new Set(["LimitationsDrawer.jsx"]);

function walkFiles(dir) {
  let files = [];
  if (!fs.existsSync(dir)) return files;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name === "archive" || entry.name === "legacy_tabs" || entry.name === "node_modules") continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files = files.concat(walkFiles(full));
    } else if (/\.(jsx?|tsx?)$/.test(entry.name)) {
      files.push(full);
    }
  }
  return files;
}

const violations = [];

const files = walkFiles(SRC_DIR);
console.log(`[AST Linter] Scanning ${files.length} source files in src/ ...`);

for (const filePath of files) {
  const fileName = path.basename(filePath);
  if (EXEMPT_FILES.has(fileName)) continue;

  const code = fs.readFileSync(filePath, "utf-8");
  let ast;
  try {
    ast = parser.parse(code, {
      sourceType: "module",
      plugins: ["jsx", "typescript"]
    });
  } catch (err) {
    console.error(`[AST Parse Error] ${filePath}: ${err.message}`);
    process.exit(1);
  }

  traverse(ast, {
    // 1. Numeric literals or metric strings in JSX text
    JSXText(p) {
      const text = p.node.value.trim();
      if (!text) return;

      // Check numeric literal in text
      if (/^\d+(\.\d+)?$/.test(text)) {
        if (!allowlist.numbers[text]) {
          violations.push({
            file: path.relative(ROOT, filePath),
            line: p.node.loc.start.line,
            literal: text,
            type: "Numeric Literal in JSX Text"
          });
        }
      } else if (METRIC_STRING_PATTERN.test(text)) {
        if (!allowlist.strings[text]) {
          violations.push({
            file: path.relative(ROOT, filePath),
            line: p.node.loc.start.line,
            literal: text,
            type: "Metric Pattern in JSX Text"
          });
        }
      }
    },

    // 2. Numbers or metric strings in Template Literals
    TemplateLiteral(p) {
      // Check quasi elements for embedded metric strings
      for (const quasi of p.node.quasis) {
        const raw = quasi.value.raw;
        const match = raw.match(METRIC_STRING_PATTERN);
        if (match && !allowlist.strings[match[0]]) {
          violations.push({
            file: path.relative(ROOT, filePath),
            line: quasi.loc.start.line,
            literal: match[0],
            type: "Metric Pattern in Template String"
          });
        }
      }
    },

    // 3. Checked props: value, label, delta, data, title, caption
    JSXAttribute(p) {
      const attrName = p.node.name?.name;
      if (!CHECKED_PROPS.has(attrName)) return;

      const valNode = p.node.value;
      if (!valNode) return;

      // Case A: string literal prop
      if (valNode.type === "StringLiteral") {
        const strVal = valNode.value.trim();
        if (/^\d+(\.\d+)?$/.test(strVal)) {
          if (!allowlist.numbers[strVal]) {
            violations.push({
              file: path.relative(ROOT, filePath),
              line: valNode.loc.start.line,
              literal: `${attrName}="${strVal}"`,
              type: `Numeric String in Prop '${attrName}'`
            });
          }
        } else if (METRIC_STRING_PATTERN.test(strVal)) {
          if (!allowlist.strings[strVal]) {
            violations.push({
              file: path.relative(ROOT, filePath),
              line: valNode.loc.start.line,
              literal: `${attrName}="${strVal}"`,
              type: `Metric String in Prop '${attrName}'`
            });
          }
        }
      }

      // Case B: JSX expression prop: value={123}
      if (valNode.type === "JSXExpressionContainer") {
        const expr = valNode.expression;
        if (expr.type === "NumericLiteral") {
          const numStr = String(expr.value);
          if (!allowlist.numbers[numStr]) {
            violations.push({
              file: path.relative(ROOT, filePath),
              line: expr.loc.start.line,
              literal: `${attrName}={${numStr}}`,
              type: `Numeric Literal in Prop '${attrName}'`
            });
          }
        } else if (expr.type === "StringLiteral") {
          const strVal = expr.value.trim();
          if (METRIC_STRING_PATTERN.test(strVal) && !allowlist.strings[strVal]) {
            violations.push({
              file: path.relative(ROOT, filePath),
              line: expr.loc.start.line,
              literal: `${attrName}={"${strVal}"}`,
              type: `Metric Expression in Prop '${attrName}'`
            });
          }
        }
      }
    }
  });
}

if (violations.length > 0) {
  console.error("\n❌ Rule 1 AST Violation - Hardcoded metric literals found in src/:");
  for (const v of violations) {
    console.error(`  ${v.file}:${v.line} [${v.type}] -> ${v.literal}`);
  }
  process.exit(1);
}

console.log("✅ check-literals passed: 0 AST violations in src/.\n");
process.exit(0);
