// Scene order. Each scene: { id, title, segs: [{ id, text, say?, pause?, pre?, silence? }], draw(ctx, T) }.
import hook from './hook.js';
import rules from './rules.js';
import search, { title as tSearch } from './search.js';
import fast, { title as tFast } from './fast.js';
import hash, { title as tHash } from './hash.js';
import sprt, { title as tSprt } from './sprt.js';
import evalsc, { title as tEval } from './evalsc.js';
import flag, { title as tFlag } from './flag.js';
import nnue, { title as tNnue } from './nnue.js';
import pack, { title as tPack } from './pack.js';
import outro, { title as tOutro } from './outro.js';
export const SCENES = [hook, rules, tSearch, search, tFast, fast, tHash, hash, tSprt, sprt, tEval, evalsc, tFlag, flag, tNnue, nnue, tPack, pack, tOutro, outro];
