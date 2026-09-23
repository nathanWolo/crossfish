#pragma once
// Compact super-board/constraint residual head. The checkpoint's embeddings
// are preprojected through the hidden layer and packed as exact float32 data.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

static const char MACRO_PACK_CJK[] = R"~(
濽滤冐贈帟氐蜀倬溏罢氠超搽燹赼剀訒漴兼恈綪笱贀炼儐媥師仡桢燽綘慀汧翴兛畨綶噫焃弇簰主瓛仪沽訍泋嘾繳竬伍壇縉紼勽害炏煕聫俄欮樍捚巁
蠗婣蝱仁巴贀笃唇捏矞楇亭维樈捓廀觪渌匤覟帰榞脁嚪薰厒茢贿厢戋戢廁崽嬼了袃繒燙蜃亶扯聺罢豙翟戃耤擁湙佬匃崕繾嵳椅蚚忯熨溇俏巇稗涤刿
劳嘫譞羵織墁椇垮砰歫眢賑娀懻紖紿憎嚛詭岙縑佩甃幍珰巽壟丄罝刊讂臀瀱塌俔丵帱牰伄炄殰歂衳代楘娔曹腀仉腄凗彐席牯挃嘈兰凈樳他拃娍峠跀
菍崌储紊縣揱焀傟聐凧潇亥赳樀紴壀录穌偯毮縓發笃脉莐嚀拻什於稂痿翀孞紌僶娜繂姖缃蛎俰櫨戇侳糜槾癎蘿屲襜咳瓳縹諊洆缴棐六只跇蔏姮垺樿
穻裬俿螭縳聚脁攻亰村咘貕媲訋怓僀蒎詳赈姨綗蛌崅挩稰也泛佭権凡嵣匿奿潄侉哺綬浜褂昃臰坙蔷亝惬娆儽瑀侌毌俊哐縔蟚仼慯堐枈娟佂螴娎湇臀
墏葔僂妜巂莫滼沠噐槺藴貉葀臹荸怿諺昛褔檃緜禭壴譃嫏溳昵俙僁戗謘槁荣櫣蝙樎繂晘壼哤児殝扁俨眀爔卾嘿斫嚃詺妈巩漂嬂溦印怆壼贛甧凲趜蝁
廇疔偃調币耸竿挐篐埯婓佅歎舕拰衁嬯蘴偎萻繽毣紆膒崰揙砯俱要爀傁叁皓剼卋填繂蛼蔇智蟰恊囧伃帶娐滔呀襻擌咈評繳潬礇裂哐簽繭倾皆般蠌敂
揦盄劇褜庙蒱茉測沰攃姻丟惍别匸楂墈恤墻纋幕惷紆怓燰汪移乱胋娒庬竁瘔炜售碬縌昜椇蕓婰搄纻伮摑爣甸繂幀攬啲簯繚卵甇撆罰庇嬝傓犒刮俼終
硾爴嗙罄幜搤挆岵洐桏赃乻盒刜安坁殄柤匂螩幹惹嬄棫幐握栥侧蝳言戞流瓥杤句沨幓嵌褂崰臰勶嵃仠灋訓噎萿汒篜乵懠縨崂愇豵痰昽恧侀俳娀肖巁
愋奓誺崥繭舁攅羂田淐楽俔勷舒箻汁蠷層吪滧帘祎朅恜蜐梋晿侶涀訐吀終跀篴千儡庚垮會翹嗰椁蛷侳伀娘趤貿愚僫賘滽綖詺苻組坰掏藮贸碪觰烰奂
坷五墋枱繌蒈僿唏盰欧窓佨梃別司孂怀纄嘪蠡帙嚒匂竖嚐匲簽丕嬽姣呼癀孌惴兑胈币膢眆廫丐檭牵了擽戉賯肾涗窔侫妻帏樲朄楓賰儏臫享呾訊竑況
痈埔凐欎繗瞰紅菉廰惈屳但便娘哗蟀嬞岼发喜帶贴愆啼姰捹恫侍怄爩荰恂徃伤坪昐庵摶缅竘匐瞇詑倷槠樐榬矁昤燔夜嬒庡痒蜋艊棐檎蠇佩咤訜艠盁
剐萜副潠繌璕複衹吰敁儋係蔙樛棡燁翤兴姏稙庁蟩唇冊曰撞瀋俭咰樐衴虂筴憤噫脘庑擼挃詵罐儨攙乐抋戙咲磀蕟筴关诖帰蜧包蔭盐拹墉侰穰爇矲晁
袯碜凿癲幪腦嬅贜癐摒儇仄粰娌濗俀嫀荜吁浑嵻炍贀谀虰奄娻但檴娒誐仂敔秬仒勯繊潴绷蚆冰泧斳予葲訝蜏偁虆娤唩曁年茾伇褲児涓贵俭訲爖罘藁
胩裫謨評庒抌圆窞彐礉丱佊厎娘垝棁賡襋赖璁縋螧贁尝欰婏皳亐祉稟拎瓀禙圼侬樄庼圀挋瞬綰殐谧丫跫稙榐珁涮乄垁圆庱褐嬈淧毐橕信侜莞戞宊赁
榩剬唚羂幰纨嬇朄児瞆亩偙訄利硅艁筬乴嘞摅繻譀茊倩侐硨溡像会娘投盁覭蒌唠议繁袣笀記氰撿毟余惜樖稛濁繆徼壌塦繧牑儈喂岐勷蟋侜跔訝珳毁
螚膴姮稻庻蘃笋訞娐签恕侻禎般蚰詂盬蚌咥癉繀苢茈俧岐笐幹傴妥戞蜊埁燯枼匴慇繀唻唄祪穰愐擃俱挑訒嬸哂璻壜労窮繄疦茈璗昐籼湃佣煬稟秫菀
徨幜姛噮庭礦謊蛋斐竽槴购漿燳萤佀埅坴傉拀縁蠰茂矴児侯獹侵倛刞孇赀罸浓諊俲帴厛贃婃昐懨筥俫葚戞杲狀謁桌儊熤縔肳甇薩像蓘糿什沰娈亴僁
矻楜咿埠繿亊夂丏芰栜渨赦橀稚炟检聂夬喣凘幄堍漇渇萐橌藵佼痉爜缫煁祆氤嗶礑繖襚挂喱材祥腡伸媕戨蕰獁艔伬卩朹繹嘹蜁氩数冋坣仫巓樅绹櫀
櫳双升掕縎矘椂裣儐窵趑佯燹稟砱磀揅筜咞襛繚蠀茈氦刐疸涉伪煢娈蚚埀嚍蚜佃倱縤僄综珩滰屺尋乣愱觠脞恀网且丶憈縝緐嬀砏惏艪嵂豟襸槬涊婀
哟繜厵稅繙奰攅罋桰楲篁亲怅娜狰懁檕犜切暲繫豯攅彍估株憗俢虛爉獹壁墾愼啴澰繚們圉区劐犿久傢禃刭洶佁勳弴壓专庞乘蔆怷彰暊庭傚蔄戭瓨歂
椌嘔匪榍繪桙蔇浶廰浺勓伶姍娕豀翁歅楼傧瞿繛萓洅蛴嬰柤绵偒臊爦翱姁茔兼啘焵繤贻褅作嶐穴滽偍渊截垐虁已憄喔场幀痈茇碈埰旨挅佭喐爜芇牁
趡帤夎忊幱謃挆响結杊皛伍檹戟忻桁于縔囒磺縃稯礂薢賰划欟侹竉臼晌品膆繼劅同繴簉椇觘筰掮嗟丸毪樗玔琿猴绌厖苴縑义愆诀臐晛唽佖傳舚蜯祁
礤覴亡秚幦娍欇筫衰坡柚跩涓舧棧嵁嘹致増妺帲觮甂蛩厰晻垖趦临姶綏县蔂碻跥娙縍狃崄繎拰侒亗仚粅刣芨偂僰幬圖壳緔璿焄楳媰斷癕偍妆利桴乂
眵氣袉巜嶏挽諻甧澐啙澧今攠燸蟕紿勊紤傛襆干綶欇漊萏潎来佃掚威压腀唕簄俤扄平嚁愅庰匰殁湗俷侙樛據湀疴而又訟繳妤褆唲现橥礳侥橄訓袖证
嚵埴丯絏繈兓脅纟猰昳晝僷戅舩剈敂兺佔囝乽幭慱猋硍呐祙宿伂坺訆袘襂翶佤塮瘬庑伇缅綋樰徢簫伂畺稆篨跁藧臬啻倻繼媠焅訿蒰楧期伿羑樖勹呂
冪罄嚖烆繘氄唄絢箰归祧伭村舢纵塂塕潄堈媥巅柪嫽糏睏繩褑亱箯討絃谿嗫紓誹譇幁獘茅確爐淍硼趥誡舐咾賀埢犄刴渼干蛈笇択皰奸沋丳涨稂僎狀
澧梴偙讟縿牪唂兠蝐勁薱仅谆訞矦烀漂潬卉庬布毎嬁槬蛐南冥业珃娰
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
    int count = d16_mini_cjk_decode(
        MACRO_PACK_CJK, buf, (int)sizeof(buf));
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
