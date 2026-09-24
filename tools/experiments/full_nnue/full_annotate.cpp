// Rewrite a search-labeled NNUEWDL1 dump so the float field is HCE + macro
// residual (both side-to-move relative). Training the local MiniNet with
// --residual on that target leaves the macro head's share alone.
#define main tb_main
#include "test_bots.cpp"
#undef main
int main(int argc, char **argv) {
    std::vector<NnueDumpPos> rows;
    if (argc < 3 || !read_nnue_dump_pos(argv[1], rows)) { fprintf(stderr, "read failed\n"); return 1; }
    auto eng = std::make_unique<CrossfishDev>();
    CrossfishDev::init_mini_lut();
    size_t bad = 0; double sum_macro = 0;
    for (auto &r : rows) {
        GlobalBoard b;
        if (!load_utttai_state(b, r.s)) { bad++; continue; }
        int full = eng->evaluate(b);
        int macro = 0;
        sum_macro += std::abs(macro);
        r.y = (float)full;
    }
    write_nnue_dump(argv[2], rows);
    printf("rows %zu bad %zu mean|macro| %.1f\n", rows.size(), bad, sum_macro / rows.size());
}
