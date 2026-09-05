"""Unit tests for tools/cg_minify.py.

These aim for 100% line coverage of cg_minify.py: tokenizer edge cases,
reserved-name handling, ice4-style renaming, token packing, and the CLI.
"""
from __future__ import annotations

import io
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import cg_minify as m  # noqa: E402


class TestIdentHelpers(unittest.TestCase):
    def test_word_char(self):
        self.assertTrue(m._is_word_char("a"))
        self.assertTrue(m._is_word_char("Z"))
        self.assertTrue(m._is_word_char("0"))
        self.assertTrue(m._is_word_char("_"))
        self.assertFalse(m._is_word_char("."))
        self.assertFalse(m._is_word_char("+"))
        self.assertFalse(m._is_word_char(" "))

    def test_is_ident(self):
        self.assertFalse(m._is_ident(""))
        self.assertFalse(m._is_ident("2foo"))
        self.assertFalse(m._is_ident("foo.bar"))
        self.assertTrue(m._is_ident("foo"))
        self.assertTrue(m._is_ident("_x"))
        self.assertTrue(m._is_ident("A1"))
        self.assertTrue(m._is_ident("mini_board"))


class TestRawStringEnd(unittest.TestCase):
    def test_r_without_quote(self):
        self.assertEqual(m._raw_string_end("R", 0), ("R", 1))
        self.assertEqual(m._raw_string_end("Rx", 0), ("R", 1))

    def test_unterminated_no_paren(self):
        src = 'R"abc'
        lex, i = m._raw_string_end(src, 0)
        self.assertEqual(lex, src)
        self.assertEqual(i, len(src))

    def test_unterminated_no_close(self):
        src = 'R"(hello'
        lex, i = m._raw_string_end(src, 0)
        self.assertEqual(lex, src)
        self.assertEqual(i, len(src))

    def test_custom_delimiter(self):
        src = 'R"MNUE(hi)MNUE";'
        lex, i = m._raw_string_end(src, 0)
        self.assertEqual(lex, 'R"MNUE(hi)MNUE"')
        self.assertEqual(src[i], ";")


class TestTokenize(unittest.TestCase):
    def test_skips_whitespace(self):
        self.assertEqual(m.tokenize("  \t\r\n int"), ["int"])

    def test_preprocessor_line_and_compact_ready(self):
        toks = m.tokenize("#include <iostream>\nint x;")
        self.assertEqual(toks[0], "#include <iostream>\n")
        self.assertIn("int", toks)

    def test_preprocessor_line_continuation(self):
        src = "#define FOO \\\n  1\nint x;"
        toks = m.tokenize(src)
        self.assertTrue(toks[0].startswith("#define FOO"))
        self.assertTrue(toks[0].endswith("\n"))
        self.assertIn("int", toks)

    def test_preprocessor_at_eof_without_newline(self):
        toks = m.tokenize("#pragma once")
        self.assertEqual(toks, ["#pragma once\n"])

    def test_preprocessor_backslash_at_eof(self):
        toks = m.tokenize("#define X \\")
        self.assertEqual(len(toks), 1)
        self.assertTrue(toks[0].startswith("#define X"))

    def test_line_comment(self):
        self.assertEqual(m.tokenize("int x; // hi\nint y;"), ["int", "x", ";", "int", "y", ";"])

    def test_line_comment_at_eof(self):
        self.assertEqual(m.tokenize("int x; // eof"), ["int", "x", ";"])

    def test_block_comment(self):
        self.assertEqual(m.tokenize("int /* c */ x;"), ["int", "x", ";"])

    def test_block_comment_unclosed(self):
        self.assertEqual(m.tokenize("int /* oops"), ["int"])

    def test_block_comment_slash_only_close(self):
        self.assertEqual(m.tokenize("int /*/ x;"), ["int"])

    def test_empty_block_comment(self):
        self.assertEqual(m.tokenize("int/**/x;"), ["int", "x", ";"])

    def test_string_literal(self):
        toks = m.tokenize(r'"hello \"world\"" foo')
        self.assertEqual(toks[0], r'"hello \"world\""')
        self.assertEqual(toks[1], "foo")

    def test_empty_string(self):
        self.assertEqual(m.tokenize('""'), ['""'])

    def test_unterminated_string(self):
        self.assertEqual(m.tokenize('"abc'), ['"abc'])

    def test_string_trailing_backslash(self):
        self.assertEqual(m.tokenize('"abc\\'), ['"abc\\'])

    def test_char_literal(self):
        toks = m.tokenize(r"char c = '\n';")
        self.assertIn(r"'\n'", toks)

    def test_char_quote_escape(self):
        self.assertIn(r"'\''", m.tokenize(r"'\''"))

    def test_empty_char_unterminated(self):
        self.assertEqual(m.tokenize("'"), ["'"])

    def test_char_trailing_backslash(self):
        self.assertEqual(m.tokenize("'\\"), ["'\\"])

    def test_raw_string_newlines_collapsed(self):
        src = 'R"MNUE(\nabc\r\ndef\n)MNUE"'
        toks = m.tokenize(src)
        self.assertEqual(len(toks), 1)
        self.assertNotIn("\n", toks[0][toks[0].find("(") + 1 : toks[0].rfind(")")])
        self.assertIn("abcdef", toks[0])

    def test_raw_string_without_paren_passthrough(self):
        toks = m.tokenize('R"abc"')
        self.assertEqual(toks[0].startswith("R"), True)

    def test_raw_string_unclosed_with_paren(self):
        toks = m.tokenize('R"(abc')
        self.assertTrue(toks[0].startswith("R\""))

    def test_r_identifier_not_raw_string(self):
        self.assertEqual(m.tokenize("R x"), ["R", "x"])
        self.assertEqual(m.tokenize("R"), ["R"])

    def test_identifiers_and_numbers(self):
        toks = m.tokenize("foo_bar 123 1.0f .5 1ull")
        self.assertEqual(toks[0], "foo_bar")
        self.assertEqual(toks[1], "123")
        self.assertEqual(toks[2], "1.0f")
        self.assertEqual(toks[3], ".5")
        self.assertEqual(toks[4], "1ull")

    def test_dot_then_ident_not_glued(self):
        self.assertEqual(m.tokenize("v.data()"), ["v", ".", "data", "(", ")"])

    def test_dot_at_eof(self):
        self.assertEqual(m.tokenize("."), ["."])

    def test_two_char_operators(self):
        src = "a==b!=c<=d>=e&&f||g<<h>>i+=j-=k*=l/=m%=n&=o|=p^=q::r->s++t--u"
        toks = m.tokenize(src)
        for op in (
            "==", "!=", "<=", ">=", "&&", "||", "<<", ">>",
            "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
            "::", "->", "++", "--",
        ):
            self.assertIn(op, toks, op)

    def test_three_char_operators(self):
        toks = m.tokenize("a<<=b>>=c...d")
        self.assertIn("<<=", toks)
        self.assertIn(">>=", toks)
        self.assertIn("...", toks)
        packed = m.minify_cpp("a<<=1; b>>=2; c...d", rename=False)
        self.assertIn("<<=", packed)
        self.assertIn(">>=", packed)

    def test_single_char_fallback(self):
        self.assertEqual(m.tokenize("@,;"), ["@", ",", ";"])

    def test_slash_not_comment(self):
        self.assertEqual(m.tokenize("a/b"), ["a", "/", "b"])
        self.assertEqual(m.tokenize("a/"), ["a", "/"])


class TestKeepAndScope(unittest.TestCase):
    def test_keep_reserved_and_intrinsics(self):
        self.assertTrue(m._keep_ident("memcpy"))
        self.assertTrue(m._keep_ident("int"))
        self.assertTrue(m._keep_ident("main"))
        self.assertTrue(m._keep_ident("__m256i"))
        self.assertTrue(m._keep_ident("_mm256_add_epi32"))
        self.assertTrue(m._keep_ident("_MM_SHUFFLE"))
        self.assertFalse(m._keep_ident("mini_board"))
        self.assertFalse(m._keep_ident("evaluate"))

    def test_follows_std_scope(self):
        toks = m.tokenize("std::chrono::milliseconds x; Foo::bar y;")
        ms = toks.index("milliseconds")
        bar = toks.index("bar")
        self.assertTrue(m._follows_std_scope(toks, ms))
        self.assertFalse(m._follows_std_scope(toks, bar))
        self.assertFalse(m._follows_std_scope(toks, 0))

    def test_after_member_op(self):
        toks = m.tokenize("v.data(); p->size(); x")
        data = toks.index("data")
        size = toks.index("size")
        x = toks.index("x")
        self.assertTrue(m._after_member_op(toks, data))
        self.assertTrue(m._after_member_op(toks, size))
        self.assertFalse(m._after_member_op(toks, x))
        self.assertFalse(m._after_member_op(toks, 0))

    def test_user_ident_at(self):
        toks = m.tokenize("int mini_board; std::max(a,b); v.data(); v.square;")
        self.assertTrue(m._user_ident_at(toks, toks.index("mini_board")))
        self.assertFalse(m._user_ident_at(toks, toks.index("int")))
        self.assertFalse(m._user_ident_at(toks, toks.index("max")))
        self.assertFalse(m._user_ident_at(toks, toks.index("data")))
        self.assertTrue(m._user_ident_at(toks, toks.index("square")))
        self.assertFalse(m._user_ident_at(toks, toks.index(";")))


class TestGenerateVariableNames(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(m.generate_variable_names(0), [])

    def test_first_names_skip_leading_underscore(self):
        names = m.generate_variable_names(3)
        self.assertEqual(names, ["a", "b", "c"])

    def test_wraps_to_two_chars(self):
        names = m.generate_variable_names(53)
        self.assertEqual(names[0], "a")
        self.assertEqual(names[51], "Z")
        self.assertEqual(names[52], "a_")
        self.assertEqual(len(set(names)), 53)
        self.assertTrue(all(n[0] != "_" for n in names))


class TestRenameIdentifiers(unittest.TestCase):
    def test_no_user_idents_returns_same(self):
        toks = ["int", ";", "return", "1"]
        self.assertEqual(m.rename_identifiers(toks), toks)
        self.assertEqual(m.rename_identifiers([]), [])

    def test_renames_long_names_consistently(self):
        src = "int mini_board; mini_board = mini_board + 1;"
        out = m.rename_identifiers(m.tokenize(src))
        joined = " ".join(out)
        self.assertNotIn("mini_board", joined)
        shorts = [t for t in out if t not in ("int", ";", "=", "+", "1")]
        self.assertTrue(shorts)
        self.assertEqual(len(set(shorts)), 1)
        self.assertLess(len(shorts[0]), len("mini_board"))

    def test_does_not_rename_one_char(self):
        out = m.stringify(m.rename_identifiers(m.tokenize("int a; a=a+1;")))
        self.assertIn("a", out)

    def test_does_not_rename_reserved_or_std(self):
        src = (
            "int main(int argc, char** argv){"
            "std::vector<int> v; v.data(); v.size();"
            "std::chrono::milliseconds ms;"
            "memcpy(a,b,n); abs(x);"
            "_mm256_set1_epi32(1); __m256i q;"
            "}"
        )
        out = m.minify_cpp(src)
        self.assertIn("main", out)
        self.assertIn("argc", out)
        self.assertIn("argv", out)
        self.assertIn("std::vector", out)
        self.assertIn(".data()", out)
        self.assertIn(".size()", out)
        self.assertIn("milliseconds", out)
        self.assertIn("memcpy", out)
        self.assertIn("abs(", out)
        self.assertIn("_mm256_set1_epi32", out)
        self.assertIn("__m256i", out)

    def test_renames_user_member_not_std_method(self):
        src = "struct Move { int mini_board; }; Move m; m.mini_board; m.data();"
        out = m.minify_cpp(src)
        self.assertNotIn("mini_board", out)
        self.assertIn(".data()", out)

    def test_arrow_std_method_kept(self):
        out = m.minify_cpp("p->size(); p->mini_board;")
        self.assertIn("->size()", out)
        self.assertNotIn("mini_board", out)

    def test_short_not_shorter_than_existing_two_char(self):
        letters = " ".join(f"int {c}=0;" for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        src = letters + " int xy=1;"
        renamed = m.rename_identifiers(m.tokenize(src))
        self.assertIn("xy", renamed)

    def test_pool_exhausted_breaks(self):
        src = "int hello_world; int another_name; int third_ident;"
        toks = m.tokenize(src)
        with mock.patch.object(m, "generate_variable_names", return_value=["int", "std"]):
            out = m.rename_identifiers(toks)
        self.assertIn("hello_world", out)
        self.assertIn("another_name", out)

    def test_match_and_mnue_literals_survive(self):
        src = 'int main(){ if(std::string(argv[1])=="match"){} const char* s=R"MNUE(blob)MNUE"; }'
        out = m.minify_cpp(src)
        self.assertIn('"match"', out)
        self.assertIn('R"MNUE(blob)MNUE"', out)
        self.assertIn("main", out)


class TestStringify(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(m.stringify([]), "")

    def test_word_space(self):
        self.assertEqual(m.stringify(["int", "x"]), "int x\n")

    def test_no_space_around_punct(self):
        self.assertEqual(m.stringify(["x", ";", "y"]), "x;y\n")

    def test_glue_pairs_get_space(self):
        self.assertEqual(m.stringify(["a", "+", "+", "b"]), "a+ +b\n")
        self.assertEqual(m.stringify(["a", "-", ">", "b"]), "a- >b\n")
        self.assertEqual(m.stringify(["a", "/", "*", "b"]), "a/ *b\n")
        self.assertEqual(m.stringify(["a", "/", "/", "b"]), "a/ /b\n")

    def test_already_endswith_newline(self):
        self.assertEqual(m.stringify(["x\n"]), "x\n")

    def test_collapses_double_newlines(self):
        self.assertEqual(m.stringify(["foo\n\n", "x"]), "foo\nx\n")

    def test_preprocessor_first(self):
        text = m.stringify(["#include <iostream>\n", "int", "x"])
        self.assertTrue(text.startswith("#include<iostream>\n"))
        self.assertIn("int x", text)

    def test_preprocessor_after_code_gets_newline(self):
        text = m.stringify(["int", "x", ";", "#include <foo>\n"])
        self.assertIn(";\n#include<foo>\n", text)

    def test_quoted_include(self):
        self.assertEqual(m._compact_pp('#include "foo.h"\n'), '#include"foo.h"\n')

    def test_pragma_not_include(self):
        self.assertEqual(m._compact_pp("#pragma once\n"), "#pragma once\n")


class TestMinifyCpp(unittest.TestCase):
    def test_strips_comments_and_indent(self):
        src = "int  foo_bar = 1; // c\n/*b*/\n"
        out = m.minify_cpp(src, rename=False)
        self.assertNotIn("//", out)
        self.assertNotIn("/*", out)
        self.assertEqual(out, "int foo_bar=1;\n")

    def test_rename_flag(self):
        src = "int foo_bar=1;"
        self.assertIn("foo_bar", m.minify_cpp(src, rename=False))
        self.assertNotIn("foo_bar", m.minify_cpp(src, rename=True))

    def test_empty_source(self):
        self.assertEqual(m.minify_cpp(""), "")

    def test_smaller_than_original(self):
        src = """
        #include <iostream>
        int mini_board_states = 0; // comment
        int evaluate_board(int mini_board_states) { return mini_board_states; }
        """
        out = m.minify_cpp(src)
        self.assertLess(len(out), len(src))
        self.assertTrue(out.endswith("\n"))


class TestMainCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_out_and_renames(self):
        src = self.dir / "in.cpp"
        dest = self.dir / "out.cpp"
        src.write_text("int foo_bar = 1;\n", encoding="utf-8")
        buf = io.StringIO()
        with mock.patch.object(sys, "argv", ["cg_minify.py", str(src), "-o", str(dest)]):
            with mock.patch("sys.stdout", buf):
                m.main()
        text = dest.read_text(encoding="utf-8")
        self.assertNotIn("foo_bar", text)
        self.assertIn("->", buf.getvalue())
        self.assertIn("saved", buf.getvalue())

    def test_default_out_is_src(self):
        src = self.dir / "in.cpp"
        src.write_text("int foo_bar = 1;\n", encoding="utf-8")
        with mock.patch.object(sys, "argv", ["cg_minify.py", str(src)]):
            with mock.patch("sys.stdout", io.StringIO()):
                m.main()
        self.assertNotIn("foo_bar", src.read_text(encoding="utf-8"))

    def test_no_rename(self):
        src = self.dir / "in.cpp"
        dest = self.dir / "out.cpp"
        src.write_text("int foo_bar = 1;\n", encoding="utf-8")
        with mock.patch.object(sys, "argv", ["cg_minify.py", str(src), "-o", str(dest), "--no-rename"]):
            with mock.patch("sys.stdout", io.StringIO()):
                m.main()
        self.assertIn("foo_bar", dest.read_text(encoding="utf-8"))

    def test_over_100k_exits(self):
        src = self.dir / "big.cpp"
        dest = self.dir / "big.min.cpp"
        src.write_text('const char* s="' + ("a" * 100001) + '";\n', encoding="utf-8")
        err = io.StringIO()
        with mock.patch.object(sys, "argv", ["cg_minify.py", str(src), "-o", str(dest)]):
            with mock.patch("sys.stdout", io.StringIO()), mock.patch("sys.stderr", err):
                with self.assertRaises(SystemExit) as cm:
                    m.main()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("100k", err.getvalue())

    def test_run_as_main(self):
        src = self.dir / "in.cpp"
        dest = self.dir / "out.cpp"
        src.write_text("int foo_bar = 1;\n", encoding="utf-8")
        argv = ["cg_minify.py", str(src), "-o", str(dest)]
        with mock.patch.object(sys, "argv", argv):
            with mock.patch("sys.stdout", io.StringIO()):
                runpy.run_path(str(TOOLS / "cg_minify.py"), run_name="__main__")
        self.assertTrue(dest.exists())
        self.assertNotIn("foo_bar", dest.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
