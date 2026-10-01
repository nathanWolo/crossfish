#pragma once
// Compact super-board/constraint residual head. The checkpoint's embeddings
// are preprojected through the hidden layer and packed as exact float32 data.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

static const char MACRO_PACK_CJK[] = R"~(
矻㞐傇㟯㠹㐄悂压㷢㯰規旼㲁㶠毄紗麭䳾㒊瓊帅搸愱埽錰脁區齳踯輐龌㪈勣㐇涶鑵劽痶㬻冀睊韠㶼偑洵痄湜廑盦䳹沁甿
鷸㛐䳠佇㕛矑鑂褴䰝㨊壮犁疹䘪还㦒鹅愒璵㡈摏劚瀂䇔䌅䡻剀鑛䡯飵禋凵㑹㴖铀贅谨䃱戏韜域䊢㠸僳㧟蘈猒驁揕逕㥀輇䝝峁
齝䘯踦㐓䪼沉鎍螦氈畩舆甶掁迠㔱抈㻶㦞㘣㵉瑵㐚萱糠㠒䠣輇屝僀谰峇鬴㐙僑㗫蕐呛䖽尙㒀䶁詓㩠森㯸䅳宆㚨鈕璍掌搀蟹䠃笂翻臀
㧦皰㲪㣞㒝㯋呹癵簑㑓㕻馀巿䅞鈘㿘铦㠢瓸鬃蒠倛凛竿蠧氿僥榜㬾暟㝢桫君桞诸駷㶺汿旯譠叾榸䃲慦㕜疕㳓䎥婙瀋堦㼃桷笿
鴅翱耼㺩嫃㐋钶逩㯂熌巹毸岀陆恽淘㶀剾㙖澛璝姘搘䷬頀斎餍蓀巎鰏陭䃄䀟㚅髐鑴㡼埶帅瘮汾㽚速箨㠨俺昆玝按䏼俬峤䋽饵蚽
务胏畦铔䋈怕㛾稛铪靳篊駀娖㥾㫤㪠嘈劊吒㤅猐獞詄䏇蕱㟺撁䄂界䧁䐱描靳緃僥㚆剹呈猯鰀跾衂勳㹠蕝孨䥓闒㣒璓輝濇瀛傑漵㿁
㖐㐔㡜䄒攟㚥繫钅音氻㮮爉䚈㝀瘍哈䋞齦㪈痎㽥摾豃栮勠佂魁秐绳峄㴏㣎鼝唰㼞谮㜯鲅敎䆡㺈壈廯㡚㥔鋷疒㷇襟䉙朄籖惁
萩㹰浵廌㖌㬛㟮蜱钰䜋谥馔昑鬸钄甁浠秸䪅滎㭚䆋甈涣蒤㠮㟹焊蟴睁蔐戬塔䁵觱㝉鐽鳒㰸樤縋豆炂璹袠䖮踊㥸愒疧哳邬贅屁
婺枰䗦䃼㙻䍃㗁㴯钙㣭篯䢭疁哰䙠柸䰽壎㨀䎙琌蛇㦖蟲䬆䧁況嬰玠䊠荽㙖哴欹簱帎渃㻓肂虜鸭矸佧㠆㑞盿㕯葔䱤䠦磗䴃韲叁
谏䍰橠琌䁿壺鰮鏛㰿毋灺鷷焐羂賽航襷鑚㲗氒皢㿼汽䌇壶壁栵㜐縯铔䙀䀡㠕渡吲噈尕腢䘂肮劀䤴䁧餈㝓㵺㝑馐琇儕䑤兠㠜諈䂷
菝菋㝖拭㑻眦咍裾䏏訅恧弴捠鯈䞨䍁鱶㥺樗畵颡䑳耔搷㝫丼滰抈㫟脓㜎钫餦谱袁䘔墄稌㴡䪦㘈慖潢㦳搨皘玊䒍倐歙嬅甤煂
鞆碐蓏蛤䮢嶭㞑钴窤薂稈礍瞂信摠肚忘厧糆㧐喅痂謌瀜炁鄋鯘䵂㟏塰汱娼㽏嘋㟚仁钇孎㱖鄓蘐鿌撄祯隠猝沨㜨戲㕁塚疑評搷揚鯮贃㬤
澖鳐已帤䆅笉㓴哵措氟䘍惙窂蒠赈䰸䁋訆㜡龃琳㖪㿪䟞闀踡瞰峎愌㵹䐃㢺蚾鐚唻啴闯阘冃嘶浸凤划㮄斄痊恠葴鹼㠟趐椇腁
暐憊靼䉝劺灕唤蘰簵緤帕癀䲂縖偠钙齘厸熥脃琯脿瞮而寑朂啚嫁鶰䧌紼㝘㧡䐅啴鼯谻䓸舁揾醃䫤䛠鎮㒈娄簲㼞阆瘟葱帏㠙骳餇竅獁
歒䖰泖䰤䈘帍㞲㠍啌䰊籋匁㘔鹅鲃榰㞡㗥粸勽呪㸈蘌皦研䒳钍䀘崫圇酮鳁頞龰哵垴㗠闣㛗钬蛜氭摮䈎㡤殅朊杠鮑嘘呸䗒㔽疜掛逞杙潂
䋐鎃霄䪿䙕㝷戻啧㼏㱔漸稍妡䞂㝎䪡㑹魈愐哲㻐鄪痮啄瀔鲚匄㓧旁䦵恰娠輌䎊蔟㙆舆镍漫氥蛊緩躄媗搡淲㘘䨸㯫瑆蒂贍頫榓焊鉂
鯫獏馰菣堅㑎䢗呑䤰䰀鸅寐亀仹挠䆈刵䏺㜗矪猲瓹䐴㼷阛㔄赁蹐烋繌㮠䪓㖅垤鐩绎氾跻樮瑺鑠吋员乾㪿䟁痼㡑搠㗺怚䉀峽胢㛁
術䴡㼬䍆䍡㘡哘哰㗶㰸胟鸋肃聊齆灈叚貎㥮齊璇貲蒭羛㠓疒褊侸奁鲨㢰彋䞜䌤䉹㒼嵝鐜悖䰝柴涀扠衵暸㞩㛫稘盋揲䑛衣弁戭见
蟝諰慴㓔䓱䱁㣷㯢咕坢䰑陨舁㳞䞀蓂䳠堋㶇耞㞞笅瑣婣䳲堌帠㤀伧㳀潾瓐㥁䍂鍣㰃毰戋榰㾂颷㛠遒佲辂㛉䐩痉刉绉倐䋅缆鏍
岦田桔艜䌗㦹㔲鉊钕牌蘛㨋咄墄示㴡䛼㘨帪㽟伂甓骉蒭䰂栧琬欆塮私硒㻐蝛甔䫍鱵㣘撁咥嫣氵乙㮃习膋骘䧹㖎㧔彯璩管蜆炘煂
嵧纐膏碼䊢瓟㞬圵铉谨䧃儠膀䢡岙㬈䟼緂㮔䙴甂牆葺蔄㤅踼癁骐珐萤䨜笩㞏萵哃杯尲红樈構纃璎鐨坊诖㐻㧵璷蕽壨膙僿䀦㫁
鬍痰堨铌䊗㙛㟷貵钭汽㭺舋馔聿䣓㩠赭劘㡀墦㫷㖞疅賧葕留愆货潀䠵朘䅴䍫控㔬䩶鏴鎓鱎龜㞜始鄠曯鄸㾎粶㨞飋玦㑨揚追鑕拿䷀
伦䃰娉餼㓉㒗㖵渔唞纀籁瀐樒䨫魾虫竜驘䮷蒪㴴逰皖芀䒩亼埪䙾錾閐䊳㪜㩒厉纼簔明忭芃锊㤹缨䠵杆㑅秦琜彰䐞嚈砜顀蜅啠䣁
㥰狠驜䆭躁㔽蜠钹䌇群僊憃穚驠鵜噸䟪懮㢋崾琋㪧耗㣼蜆㩬䭂㧐蘢啴䑯呕㡮瑽哚臆屝壑縕詍鎂㘥侩筨恽璲㹮葘癄㰾䑛䀑蕜弄㤻黀
榽䏰濳䋶㳯㟡縈钾䏶谶魤爉㾂锾圡㞪隈噚䨶㦇璃甯敛摅奛倒鼪儈豚繂䢫㤐葁繓莥厃宱㰖佫捃厮羟峨䑥恪㦪痼菩牐䅽茔籁
㪐䌐浱粴䊤㕯㮢鐙鎨谄㼺娄乶誁䩽菠玑蠸㰶㬪㔰濘瓅茴怈呐輅墡恀㫮泐㽽舴㚫瞥㐵䬌門
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

__attribute__((always_inline)) static inline int evaluate_macro_key(int constraint, int key) {
    if (!MACRO_READY && !macro_load_packed()) return 0;
    return MACRO_SCORE[constraint][key];
}

