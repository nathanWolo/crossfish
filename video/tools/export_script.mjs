// Dump the narration of every scene to build/script.json (for tts.py) and SCRIPT.md (for people).
import fs from 'node:fs';
import { SCENES } from '../src/scenes/index.js';
fs.mkdirSync('build', { recursive: true });
const segs = [];
let md = '# Crossfish explainer: narration script\n\nGenerated from `src/scenes/*.js` by `tools/export_script.mjs`. Edit the scenes, not this file.\n';
let words = 0;
for (const sc of SCENES) {
  const spoken = sc.segs.filter(s => !s.silence);
  if (spoken.length) md += `\n## ${sc.title ?? sc.id}\n\n`;
  for (const s of spoken) {
    segs.push({ key: `${sc.id}.${s.id}`, text: s.text, say: s.say ?? s.text });
    md += `${s.text}\n\n`; words += s.text.split(/\s+/).length;
  }
}
fs.writeFileSync('build/script.json', JSON.stringify(segs, null, 1));
fs.writeFileSync('SCRIPT.md', md.replace('Edit the scenes, not this file.', `Edit the scenes, not this file. ${words} words.`));
console.log(`${segs.length} segments, ${words} words`);
