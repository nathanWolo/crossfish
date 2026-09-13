#pragma once
// Compact super-board/constraint residual head. The checkpoint's embeddings
// are preprojected through the hidden layer and packed as exact float32 data.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

static const char MACRO_PACK_B85[] = R"~(
L\;3Cj?Eo8IGkZ$!\Apf(+4o&Kin%5TAKc:n4F?Zh\&JoWm`'*,.n=Lb#A_`iG"06gZB/7H(Q0Df&p_\fS6oOd0I1s"1bkUl?`3?
f<WI]_9U?A6V>D7-M,&<9s]^@2-cTd`,rQ*eQ.aIkZl+k>X6+ie*m`1`s5U.S`\>8Xi&Q%KiHF7n#?B(8NWP0_7ApJ8?1O^=Y31Z
58K)@[0Vl14S7+<+hE<B8j/b2^p[+T-`gFa=iYqU4GaiiJRfP45gL1]q"H\_Ga_.:3f"rqcT`JeGNf5p'$PsB=0]0L:oG/Gaf_SY
f"+Ddd4bSc]>^s<:,n*F)bug:6l)3+cqTkBs3eBl"N>IJY@+]PLr?+)FT=9a`^^"p,(t-Ue1R9,j)&2?7QtlY"",ojp!HT$[RTpT
a9W6j4GO?\el*!me$qWXf/3!4W\4^pNF*O<!h[S"5'Tr?V>\ZS$dB$o797@BS)*MJ:5@"Xo*Nu5L]ouX^g?TI.KAqR21"c-]o3rI
#eb%>ps',\c@l;8Zb(4ll4^Zl3QnhHXQK,ikKKj?Yr$5]2UQica4rekibk8NZ1"j2J,-4^cR#[.=$$K"PnBlWPqpOu["Q<3g,m>>
dL59Q?];n=<@P/f"o#IAD/c:7'A18g/cHGL@Rg4.Z1ZDXQik[MQ(?18SP-3&@*`FM"u4oKEe_rV;k9RG'5o9bR"GX[6KZY1kRmF%
.R.K0Y`##9$X1&)*hXZB^TIGuM:htqSR)pnoBCh?`>jNFEk9]44[V"h$T8G[fa@sLImtBneW%j<<a6-N$&*uG#364Zg!pX"kbJ8G
6!oF4?EArI*fGh=V@Xt;TV`I?+C`)M)k<E_0BY>]6QUM<OSljY#:#-keI]2TT(<Z47<L[X<BN<.1r"Z.PI<Qhp"hdHZ][Ge?(-\f
na6TH!2)csU.e(n;T`dk$]iQKfQu_s,bd<8"$kMT#W2fOmHn`+U-)Glc5+R=qSF1<j4kS?5,&u_c@Zp"=.5(6+o<6GB58MS?n?W?
E&.F.+%nFWYtc6).Np-*Y4.?8<A/^8OqWnmk,SErEQ]oL,V(F^TMd5TbGqYZ0nI?\pJWf_?q`2h/m1Y13R0<F5ll#Bgo`<HBNJ?Q
eqV/[#]iL&PEJI'i.!CHWVStJp2NJdEFM@&NjX7#V#SfWH_X+t_dZ2'FqL<35e3j[jY)f`)XB[T?5AA`Vrc7?"UCYcS!B:*H![F8
9_?WGO"enq:Lhbei/','1)%)\GtQo.*E1n3T^X5CrcOjLK@H=*J-X<6%&K$$9J!S9/rq<p[m1\EEIs^sr<D.Fe.^_*.rO30khP;d
MP0,t@^P8`;I&9KW4!ea[XJ=p!IugirrV(D:r=GCBU\'KnbF9CYT>sDZNL9EO6JjC9P?&n[NPO_>K<j0mmJJ<=T/-333'$o#eUO3
-/8enU=lff1MD<R`t+r]dX*7UBPIrB;18s@8-9,a$G%BGA;MHpXO\R\;_*no'f>Qs0qTjZ2)-aLe8iVq#<F>#OQum4b1Re#L0O=9
T=a[.IU:,]C<O4S4i8U4)cQXL0`:d_Wu:fMX#+/!T-0[ioGnFX?CM>=Fid"e/;'J/7L&e,J[P<h1jJ?&F@Vidg[dhLLO:bk=-T/i
NCX[KSQldo7LTnr>*ojfeRr\aCtKFgR"#f:2fF8`#4!_S@$3Z>`siYJL4([@oL`mbaR4VqBc<2ObnqkF&OU!C)rY2@121(2qKWpo
F:c=*_VA_=[^]h`a,i5HoB[>]#G.'o_hqs?C$:/bm2EgW%\5RG[F%$%)Y4e47;p.1r,otD153_5:f8l2OfYRuh1!s@h#5eh_if"*
G:fRU:QhG<Yd)&1H@2)GlFs75s6+JhX#=HRIiP.JAK\h)`4Eb"aE>>71BUXnJ6u/:mO6u.4ojQWZpP>>7WQkl0-gP7>uJDP1kb>6
.MkcApkB9pmWIm3LtTtfhF#.Gi>BEtb#lJGdo50jq_AUHq*3+imI87aCfWL2bGVMX8j$*?8P$]aXI_jIY`K8Z8esD&JnJ9'qhMPn
eY[o50:IC+:*"Gl`;2%#OKBkGoeWJ\Y%3a3CE*IDn;mf?l)_V8faiqT/,ucgQK0@.I^fMI?qCPha0>_5+6Ri6`DCeFSCVsOj'obh
DPTb<`j(+]_[;#^f`VaNQs_C1=muthRZV\X[BKm&#F7fpn'he5JggmjVMHWecNIptKdCEKm^HlGa??]B#HBjN4;`=ZC\N%h=\EbN
CBib6_&43/ZO1<\Sja4-#D,!%:O[?#SGuO(fmMVlk%d%FqlDkGmsmJdr#ROYfAr2_]<44^c7D(EUU,cmWG"l>"&:F7!TuL1d0(LW
Zn@V0mQucHO1h%!9bU%5!->E].r)?*0a4etnPi#p,>JH[Rd@/m/M+'@&peC]-C8+,kK"4N@u[e1qYKm#7\L"&p;h,<[^'g<\BA2F
cdr%Z/VO*f4c'2)-PVn4U/\qD#GV_nWUpWb)Ym]*[0US7j&,hA`SpJS14f!]crn-!bX5c(5)WQ0le,u<k$4^toED&+_P<pB;_'lE
pSeNbD2Z0E^j<f3J1Ui=VsUn.YK>_=r!9%4J.L)ETu[R*AoRmgW2J@baU#2gkel%=&>7ZAi!7n#`6J6HI%M`ObY$n]XVp+arMpOH
>IC:mLTtDW[TqHN3&4K#NSGgGd81Il>$3l%jFa$h?7B0q33lZ%lT\+Z>WuE.F3In*;:UrI3h;#@]&O2cb?s<rDia%sqM]1)<u0?&
Vn60LCS2,'*[VFbH\^TH.ZK8"N[BUu+ub@o:r4s&O=X'i-0^&tkG*XE@naqW%*+Mr`C'_l%?lU`G5VM..M#0:7\e>E?gZgda5W%C
)h/%-D/9BlA@D`YED(#^[juD_Z:G\GaZL6QE`PcJ!SF]oYNGMo;,%20U86neio+c?R9iM%.etm:h''#=BH\!T762&J9CNN[U94nk
f?Jk4'?q1W[(m>AGQj)COuGhGLnlOfRbSKAft&1jCbUE+8t1gVWMQ0%hf_B'pRsF[2D6o'?]["YFVVC$WTh16DdbT@Gk<i/SkM:6
;U6gZCj\mle5/*?eg!&gPOlN)6!QT?a%/E]9n4ZZU1?5L5KPlNjU!t6!dotnlA-h(IYC'RD&u#0bou.6re1D?_i`)-a3Z\0L<[Ct
Pc?(H%<CProR>L[!]pEL1ogDf%`8(X\"`c85po1=\[*YJ`Y/Qmc5R#)+[umVdbpG-qq/<m>2f4k3-V@<H"A<ePg^&WA[M*F6j/aj
]thr[KsAri<XR/=Y"S(HXHU7]-u7O&R)%<>m;(t%Q54YYl$cDpG.Gpb+QksUV9?G8bJ*5V\EOdFLU(L+V?"uXCZECKg^:*sq+M[A
2G>pC0MYQ^)'C6@g:@tb4XpM"$c*%a7>g]O;JZ'D)fmdNH[6Dn9eI8LQCS"BUt+HgU/0Q.4]C6%k9l!LFdQmYb$9N_Ncf8)'.WQ<
VY`hKb55g;i<n4$']rOa6WRm9&Rt>s**1u_Gn`Y4b_QtL:p:YMfGG@>pBG3F[C2_nT$nguE]bG[bdpQcLGG13``gU&V/[0b5,Gs/
,1N5M$h$m-+>l9tE1`B2K/ON4pJ;YmOV/Hs%HnNn2aDdW[5Ssil:i-_[?GOF![[0JDPfK^a4M.oS#@l)6kF`3Di8SN)Cp]]!hJVl
CO/4]hk**KhSM@jnB)$QHp??$[WY0W5K6..@6R.I%]^:1-)/Dq@GjGf04MB+'e7nQ/S.F=/1jI\.97-LJdeJ?'qV0bDU5.D#+iau
Z*qE_ld5Vq+0P,u/q$5U#$$XKC:]Kmr7[c-V=KC-!O'5Q-]1G\%3Rc!bD_?Fe=NDM[tg]n1Oc(6+L@lZL7PA1Q@5C@r^NTsK4*e+
9HEIb`V:Cgl?T,>KF/L2;GaG[%G]u/>O:B8<H.1f)OClM
)~";

alignas(32) static float MACRO_BASE[16];
alignas(32) static float MACRO_CONSTR[10][16];
alignas(32) static float MACRO_EMB[9][4][16];
alignas(32) static float MACRO_OUT[16];
static float MACRO_BIAS = 0;
static bool MACRO_READY = false;
static constexpr int MACRO_CLIP = 2000;
static constexpr int MACRO_KEY_STATES = 1 << 18;
alignas(64) static int16_t MACRO_SCORE[10][MACRO_KEY_STATES];

static int macro_finish_hidden(__m256 h0, __m256 h1) {
    const __m256 zero = _mm256_setzero_ps();
    h0 = _mm256_max_ps(h0, zero);
    h1 = _mm256_max_ps(h1, zero);
    h0 = _mm256_mul_ps(h0, _mm256_load_ps(MACRO_OUT));
    h1 = _mm256_mul_ps(h1, _mm256_load_ps(MACRO_OUT + 8));
    int value = (int)std::lround(
        MACRO_BIAS + d16_mini_hsum256(_mm256_add_ps(h0, h1)));
    return std::max(-MACRO_CLIP, std::min(MACRO_CLIP, value));
}

static bool macro_load_packed() {
    if (MACRO_READY) return true;
    static unsigned char buf[4096];
    int count = d16_mini_b85_decode(
        MACRO_PACK_B85, buf, (int)sizeof(buf));
    const int need = (16 + 10 * 16 + 9 * 4 * 16 + 16 + 1) * 4;
    if (count < need) return false;
    int off = 0;
    memcpy(MACRO_BASE, buf + off, sizeof(MACRO_BASE));
    off += sizeof(MACRO_BASE);
    memcpy(MACRO_CONSTR, buf + off, sizeof(MACRO_CONSTR));
    off += sizeof(MACRO_CONSTR);
    memcpy(MACRO_EMB, buf + off, sizeof(MACRO_EMB));
    off += sizeof(MACRO_EMB);
    memcpy(MACRO_OUT, buf + off, sizeof(MACRO_OUT));
    off += sizeof(MACRO_OUT);
    memcpy(&MACRO_BIAS, buf + off, sizeof(MACRO_BIAS));
    for (int constraint = 0; constraint < 10; constraint++) {
        for (int key = 0; key < MACRO_KEY_STATES; key++) {
            __m256 h0 = _mm256_add_ps(
                _mm256_load_ps(MACRO_BASE),
                _mm256_load_ps(MACRO_CONSTR[constraint]));
            __m256 h1 = _mm256_add_ps(
                _mm256_load_ps(MACRO_BASE + 8),
                _mm256_load_ps(MACRO_CONSTR[constraint] + 8));
            int packed = key;
            for (int mb = 0; mb < 9; mb++, packed >>= 2) {
                int cls = packed & 3;
                h0 = _mm256_add_ps(
                    h0, _mm256_load_ps(MACRO_EMB[mb][cls]));
                h1 = _mm256_add_ps(
                    h1, _mm256_load_ps(MACRO_EMB[mb][cls] + 8));
            }
            MACRO_SCORE[constraint][key] =
                (int16_t)macro_finish_hidden(h0, h1);
        }
    }
    MACRO_READY = true;
    return true;
}

static int evaluate_macro_key(int constraint, int key) {
    if (!MACRO_READY && !macro_load_packed()) return 0;
    return MACRO_SCORE[constraint][key];
}

template <typename Board>
static int evaluate_macro_fast(const Board &board) {
    if (!MACRO_READY && !macro_load_packed()) return 0;
    const int stm = board.n_moves & 1;
    const int constraint = d16_mini_board_constraint(board);
    __m256 h0 = _mm256_add_ps(
        _mm256_load_ps(MACRO_BASE),
        _mm256_load_ps(MACRO_CONSTR[constraint]));
    __m256 h1 = _mm256_add_ps(
        _mm256_load_ps(MACRO_BASE + 8),
        _mm256_load_ps(MACRO_CONSTR[constraint] + 8));
    for (int mb = 0; mb < 9; mb++) {
        const int bit = 1 << mb;
        int cls = 0;
        if (board.mini_board_states[stm] & bit) cls = 1;
        else if (board.mini_board_states[stm ^ 1] & bit) cls = 2;
        else if (board.mini_board_states[2] & bit) cls = 3;
        h0 = _mm256_add_ps(
            h0, _mm256_load_ps(MACRO_EMB[mb][cls]));
        h1 = _mm256_add_ps(
            h1, _mm256_load_ps(MACRO_EMB[mb][cls] + 8));
    }
    return macro_finish_hidden(h0, h1);
}
