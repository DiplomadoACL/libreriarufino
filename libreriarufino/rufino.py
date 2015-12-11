# -*- coding: utf-8 -*-
import urllib2
import bz2
import re
import numpy
from nltk.corpus import wordnet as wn

def monge_elkan(A,B,lexical_similarity,exponent=1):
    summation=0.0
    for a in A:
    	max_similarity_versus_a=0.0
    	for b in B:
    	    similarity=(lexical_similarity(a,b))**exponent
    	    if similarity>max_similarity_versus_a:
    	    	max_similarity_versus_a=similarity
    	summation+=max_similarity_versus_a
    return (float(summation)/len(A))**(1.0/exponent)
    	    


# knowledge-based lexical similarity functions
# 1. path similarity
def path_similarity(a,b):
    synsets_a=wn.synsets(a)
    synsets_b=wn.synsets(b)
    max_sim_path=0
    for synset_in_a in synsets_a:
    	for synset_in_b in synsets_b:
    	   sim_path_length=synset_in_a.path_similarity(synset_in_b)
    	   if sim_path_length>max_sim_path:
    	   	max_sim_path=sim_path_length
    return max_sim_path
    
    


# Edit distance
def edit_distance(a,b,del_cost=lambda x:1,ins_cost=lambda x:1,subs_cost=lambda x,y:1):
    D=numpy.zeros((len(a)+1,len(b)+1))
    for i in range(1,len(a)+1):
        D[i,0]=i
        for j in range(1,len(b)+1):
            D[0,j]=j
            if a[i-1]==b[j-1]:
                D[i,j]=D[i-1,j-1]
            else:
                D[i,j]=min(D[i-1,j]+del_cost(a[i-1]),# 1 costo borrado
                           D[i,j-1]+ins_cost(b[j-1]), # 1 costo insercion
                           D[i-1,j-1]+subs_cost(a[i-1],b[j-1]))# 1 es costo reemplazo
    return D[len(a),len(b)]

# WORD TOKENIZER
def split_words(text):
    SPECIAL_CHARACTERS=u",.;:(){}[]+*¡!¿?&%$#'/£~|="+'"'
    for char in SPECIAL_CHARACTERS:
        #text=text.decode("utf-8").replace(char,u" ")
        text=text.replace(char," ")
    return text.split()
    
# SENTENCE SPLITER
# BY: Alfredo Morales
def split_sentences(text):
    for sentence_separator in [u'. ',u'.\n',u'? ',u'! ',u'?\n',u'!\n',u'; ',u';\n',u'- ',u'--',u'...',u'\n',u'\n\n',u'\n\n\n']:
	text=text.replace(sentence_separator,u'|||')
    return text.split(u'|||')
    
    
# SENTENCE TOKENIZER
def split_sentences(text):
	for sentence_separator in [u'. ',u'.\n',u'? ',u'! ',u'?\n',u'!\n',u'; ',u';\n',u'- ',u'--',u'...',u'\n',u'\n\n',u'\n\n\n']:
		text=text.replace(sentence_separator,u'|||')
		return text.split(u'|||')


# ISO LANGUAGES CODES AND NAMES
ISO_SPANISH={"aa":u"afar",	"ab":u"abjasio (o abjasiano)",	"ae":u"avéstico",	"af":u"afrikáans",	"ak":u"akano",	"am":u"amhárico",	"an":u"aragonés",	"ar":u"árabe",	"as":u"asamés",	"av":u"avar (o ávaro)",	"ay":u"aimara",	"az":u"azerí",	"ba":u"baskir",	"be":u"bielorruso",	"bg":u"búlgaro",	"bh":u"bhoyapurí",	"bi":u"bislama",	"bm":u"bambara",	"bn":u"bengalí",	"bo":u"tibetano",	"br":u"bretón",	"bs":u"bosnio",	"ca":u"catalán",	"ce":u"checheno",	"ch":u"chamorro",	"co":u"corso",	"cr":u"cree",	"cs":u"checo",	"cu":u"eslavo eclesiástico antiguo",	"cv":u"chuvasio",	"cy":u"galés",	"da":u"danés",	"de":u"alemán",	"dv":u"maldivo (o dhivehi)",	"dz":u"dzongkha",	"ee":u"ewé",	"el":u"griego (moderno)",	"en":u"inglés",	"eo":u"esperanto",	"es":u"español",	"et":u"estonio",	"eu":u"euskera",	"fa":u"persa",	"ff":u"fula",	"fi":u"finés (o finlandés)",	"fj":u"fiyiano (o fiyi)",	"fo":u"feroés",	"fr":u"francés",	"fy":u"frisón (o frisio)",	"ga":u"irlandés (o gaélico)",	"gd":u"gaélico escocés",	"gl":u"gallego",	"gn":u"guaraní",	"gu":u"guyaratí (o guyaratí)",	"gv":u"manés (gaélico manés o de Isla de Man)",	"ha":u"hausa",	"he":u"hebreo",	"hi":u"hindi (o hindú)",	"ho":u"hiri motu",	"hr":u"croata",	"ht":u"haitiano",	"hu":u"húngaro",	"hy":u"armenio",	"hz":u"herero",	"ia":u"interlingua",	"id":u"indonesio",	"ie":u"occidental",	"ig":u"igbo",	"ii":u"yi de Sichuán",	"ik":u"iñupiaq",	"io":u"ido",	"is":u"islandés",	"it":u"italiano",	"iu":u"inuktitut (o inuit)",	"ja":u"japonés",	"jv":u"javanés",	"ka":u"georgiano",	"kg":u"kongo (o kikongo)",	"ki":u"kikuyu",	"kj":u"kuanyama",	"kk":u"kazajo (o kazajio)",	"kl":u"groenlandés (o kalaallisut)",	"km":u"camboyano (o jemer)",	"kn":u"canarés",	"ko":u"coreano",	"kr":u"kanuri",	"ks":u"cachemiro (o cachemir)",	"ku":u"kurdo",	"kv":u"komi",	"kw":u"córnico",	"ky":u"kirguís",	"la":u"latín",	"lb":u"luxemburgués",	"lg":u"luganda",	"li":u"limburgués",	"ln":u"lingala",	"lo":u"lao",	"lt":u"lituano",	"lu":u"luba-katanga (o chiluba)",	"lv":u"letón",	"mg":u"malgache (o malagasy)",	"mh":u"marshalés",	"mi":u"maorí",	"mk":u"macedonio",	"ml":u"malayalam",	"mn":u"mongol",	"mr":u"maratí",	"ms":u"malayo",	"mt":u"maltés",	"my":u"birmano",	"na":u"nauruano",	"nb":u"noruego bokmål",	"nd":u"ndebele del norte",	"ne":u"nepalí",	"ng":u"ndonga",	"nl":u"neerlandés",	"nn":u"nynorsk",	"no":u"noruego",	"nr":u"ndebele del sur",	"nv":u"navajo",	"ny":u"chichewa",	"oc":u"occitano",	"oj":u"ojibwa",	"om":u"oromo",	"or":u"oriya",	"os":u"osético (u osetio, u oseta)",	"pa":u"panyabí (o penyabi)",	"pi":u"pali",	"pl":u"polaco",	"ps":u"pastú (o pastún, o pashto)",	"pt":u"portugués",	"qu":u"quechua",	"rm":u"romanche",	"rn":u"kirundi",	"ro":u"rumano",	"ru":u"ruso",	"rw":u"ruandés (o kiñaruanda)",	"sa":u"sánscrito",	"sc":u"sardo",	"sd":u"sindhi",	"se":u"sami septentrional",	"sg":u"sango",	"si":u"cingalés",	"sk":u"eslovaco",	"sl":u"esloveno",	"sm":u"samoano",	"sn":u"shona",	"so":u"somalí",	"sq":u"albanés",	"sr":u"serbio",	"ss":u"suazi (o swati, o siSwati)",	"st":u"sesotho",	"su":u"sundanés (o sondanés)",	"sv":u"sueco",	"sw":u"suajili",	"ta":u"tamil",	"te":u"télugu",	"tg":u"tayiko",	"th":u"tailandés",	"ti":u"tigriña",	"tk":u"turcomano",	"tl":u"tagalo",	"tn":u"setsuana",	"to":u"tongano",	"tr":u"turco",	"ts":u"tsonga",	"tt":u"tártaro",	"tw":u"twi",	"ty":u"tahitiano",	"ug":u"uigur",	"uk":u"ucraniano",	"ur":u"urdu",	"uz":u"uzbeko",	"ve":u"venda",	"vi":u"vietnamita",	"vo":u"volapük",	"wa":u"valón",	"wo":u"wolof",	"xh":u"xhosa",	"yi":u"yídish (o yidis, o yiddish)",	"yo":u"yoruba",	"za":u"chuan (o chuang, o zhuang)",	"zh":u"chino",	"zu":u"zulú",}
ISO_ENGLISH={"ab":"Abkhaz",	"aa":"Afar",	"af":"Afrikaans",	"ak":"Akan",	"sq":"Albanian",	"am":"Amharic",	"ar":"Arabic",	"an":"Aragonese",	"hy":"Armenian",	"as":"Assamese",	"av":"Avaric",	"ae":"Avestan",	"ay":"Aymara",	"az":"Azerbaijani",	"bm":"Bambara",	"ba":"Bashkir",	"eu":"Basque",	"be":"Belarusian",	"bn":"Bengali, Bangla",	"bh":"Bihari",	"bi":"Bislama",	"bs":"Bosnian",	"br":"Breton",	"bg":"Bulgarian",	"my":"Burmese",	"ca":"Catalan",	"ch":"Chamorro",	"ce":"Chechen",	"ny":"Chichewa, Chewa, Nyanja",	"zh":"Chinese",	"cv":"Chuvash",	"kw":"Cornish",	"co":"Corsican",	"cr":"Cree",	"hr":"Croatian",	"cs":"Czech",	"da":"Danish",	"dv":"Divehi, Dhivehi, Maldivian",	"nl":"Dutch",	"dz":"Dzongkha",	"en":"English",	"eo":"Esperanto",	"et":"Estonian",	"ee":"Ewe",	"fo":"Faroese",	"fj":"Fijian",	"fi":"Finnish",	"fr":"French",	"ff":"Fula, Fulah, Pulaar, Pular",	"gl":"Galician",	"ka":"Georgian",	"de":"German",	"el":"Greek (modern)",	"gn":"Guaraní",	"gu":"Gujarati",	"ht":"Haitian, Haitian Creole",	"ha":"Hausa",	"he":"Hebrew",	"hz":"Herero",	"hi":"Hindi",	"ho":"Hiri Motu",	"hu":"Hungarian",	"ia":"Interlingua",	"id":"Indonesian",	"ie":"Interlingue",	"ga":"Irish",	"ig":"Igbo",	"ik":"Inupiaq",	"io":"Ido",	"is":"Icelandic",	"it":"Italian",	"iu":"Inuktitut",	"ja":"Japanese",	"jv":"Javanese",	"kl":"Kalaallisut, Greenlandic",	"kn":"Kannada",	"kr":"Kanuri",	"ks":"Kashmiri",	"kk":"Kazakh",	"km":"Khmer",	"ki":"Kikuyu, Gikuyu",	"rw":"Kinyarwanda",	"ky":"Kyrgyz",	"kv":"Komi",	"kg":"Kongo",	"ko":"Korean",	"ku":"Kurdish",	"kj":"Kwanyama, Kuanyama",	"la":"Latin",	"lb":"Luxembourgish, Letzeburgesch",	"lg":"Ganda",	"li":"Limburgish, Limburgan, Limburger",	"ln":"Lingala",	"lo":"Lao",	"lt":"Lithuanian",	"lu":"Luba-Katanga",	"lv":"Latvian",	"gv":"Manx",	"mk":"Macedonian",	"mg":"Malagasy",	"ms":"Malay",	"ml":"Malayalam",	"mt":"Maltese",	"mi":"Māori",	"mr":"Marathi (Marāṭhī)",	"mh":"Marshallese",	"mn":"Mongolian",	"na":"Nauru",	"nv":"Navajo, Navaho",	"nd":"Northern Ndebele",	"ne":"Nepali",	"ng":"Ndonga",	"nb":"Norwegian Bokmål",	"nn":"Norwegian Nynorsk",	"no":"Norwegian",	"ii":"Nuosu",	"nr":"Southern Ndebele",	"oc":"Occitan",	"oj":"Ojibwe, Ojibwa",	"cu":"Old Church Slavonic, Church Slavonic, Old Bulgarian",	"om":"Oromo",	"or":"Oriya",	"os":"Ossetian, Ossetic",	"pa":"Panjabi, Punjabi",	"pi":"Pāli",	"fa":"Persian (Farsi)",	"pl":"Polish",	"ps":"Pashto, Pushto",	"pt":"Portuguese",	"qu":"Quechua",	"rm":"Romansh",	"rn":"Kirundi",	"ro":"Romanian",	"rh":"Rothongua",	"ru":"Russian",	"sa":"Sanskrit (Saṁskṛta)",	"sc":"Sardinian",	"sd":"Sindhi",	"se":"Northern Sami",	"sm":"Samoan",	"sg":"Sango",	"sr":"Serbian",	"gd":"Scottish Gaelic, Gaelic",	"sn":"Shona",	"si":"Sinhala, Sinhalese",	"sk":"Slovak",	"sl":"Slovene",	"so":"Somali",	"st":"Southern Sotho",	"es":"Spanish",	"su":"Sundanese",	"sw":"Swahili",	"ss":"Swati",	"sv":"Swedish",	"ta":"Tamil",	"te":"Telugu",	"tg":"Tajik",	"th":"Thai",	"ti":"Tigrinya",	"bo":"Tibetan Standard, Tibetan, Central",	"tk":"Turkmen",	"tl":"Tagalog",	"tn":"Tswana",	"to":"Tonga (Tonga Islands)",	"tr":"Turkish",	"ts":"Tsonga",	"tt":"Tatar",	"tw":"Twi",	"ty":"Tahitian",	"ug":"Uyghur",	"uk":"Ukrainian",	"ur":"Urdu",	"uz":"Uzbek",	"ve":"Venda",	"vi":"Vietnamese",	"vo":"Volapük",	"wa":"Walloon",	"cy":"Welsh",	"wo":"Wolof",	"fy":"Western Frisian",	"xh":"Xhosa",	"yi":"Yiddish",	"yo":"Yoruba",	"za":"Zhuang, Chuang",	"zu":"Zulu",}
ISO_NATIVE={"ab":u"аҧсуа бызшәа, аҧсшәа",	"aa":u"Afaraf",	"af":u"Afrikaans",	"ak":u"Akan",	"sq":u"Shqip",	"am":u"አማርኛ",	"ar":u"العربية",	"an":u"aragonés",	"hy":u"Հայերեն",	"as":u"অসমীয়া",	"av":u"авар мацӀ, магӀарул мацӀ",	"ae":u"avesta",	"ay":u"aymar aru",	"az":u"azərbaycan dili",	"bm":u"bamanankan",	"ba":u"башҡорт теле",	"eu":u"euskara, euskera",	"be":u"беларуская мова",	"bn":u"বাংলা",	"bh":u"भोजपुरी",	"bi":u"Bislama",	"bs":u"bosanski jezik",	"br":u"brezhoneg",	"bg":u"български език",	"my":u"ဗမာစာ",	"ca":u"català",	"ch":u"Chamoru",	"ce":u"нохчийн мотт",	"ny":u"chiCheŵa, chinyanja",	"zh":u"中文 (Zhōngwén), 汉语, 漢語",	"cv":u"чӑваш чӗлхи",	"kw":u"Kernewek",	"co":u"corsu, lingua corsa",	"cr":u"ᓀᐦᐃᔭᐍᐏᐣ",	"hr":u"hrvatski jezik",	"cs":u"čeština, český jazyk",	"da":u"dansk",	"dv":u"ދިވެހި",	"nl":u"Nederlands, Vlaams",	"dz":u"རྫོང་ཁ",	"en":u"English",	"eo":u"Esperanto",	"et":u"eesti, eesti keel",	"ee":u"Eʋegbe",	"fo":u"føroyskt",	"fj":u"vosa Vakaviti",	"fi":u"suomi, suomen kieli",	"fr":u"français, langue française",	"ff":u"Fulfulde, Pulaar, Pular",	"gl":u"galego",	"ka":u"ქართული",	"de":u"Deutsch",	"el":u"ελληνικά",	"gn":u"Avañe'ẽ",	"gu":u"ગુજરાતી",	"ht":u"Kreyòl ayisyen",	"ha":u"(Hausa) هَوُسَ",	"he":u"עברית",	"hz":u"Otjiherero",	"hi":u"हिन्दी, हिंदी",	"ho":u"Hiri Motu",	"hu":u"magyar",	"ia":u"Interlingua",	"id":u"Bahasa Indonesia",	"ie":u"Originally called Occidental; then Interlingue after WWII",	"ga":u"Gaeilge",	"ig":u"Asụsụ Igbo",	"ik":u"Iñupiaq, Iñupiatun",	"io":u"Ido",	"is":u"Íslenska",	"it":u"italiano",	"iu":u"ᐃᓄᒃᑎᑐᑦ",	"ja":u"日本語 (にほんご)",	"jv":u"basa Jawa",	"kl":u"kalaallisut, kalaallit oqaasii",	"kn":u"ಕನ್ನಡ",	"kr":u"Kanuri",	"ks":u"कश्मीरी, كشميري‎",	"kk":u"қазақ тілі",	"km":u"ខ្មែរ, ខេមរភាសា, ភាសាខ្មែរ",	"ki":u"Gĩkũyũ",	"rw":u"Ikinyarwanda",	"ky":u"Кыргызча, Кыргыз тили",	"kv":u"коми кыв",	"kg":u"Kikongo",	"ko":u"한국어, 조선어",	"ku":u"Kurdî, كوردی‎",	"kj":u"Kuanyama",	"la":u"latine, lingua latina",	"lb":u"Lëtzebuergesch",	"lg":u"Luganda",	"li":u"Limburgs",	"ln":u"Lingála",	"lo":u"ພາສາລາວ",	"lt":u"lietuvių kalba",	"lu":u"Tshiluba",	"lv":u"latviešu valoda",	"gv":u"Gaelg, Gailck",	"mk":u"македонски јазик",	"mg":u"fiteny malagasy",	"ms":u"bahasa Melayu, بهاس ملايو‎",	"ml":u"മലയാളം",	"mt":u"Malti",	"mi":u"te reo Māori",	"mr":u"मराठी",	"mh":u"Kajin M̧ajeļ",	"mn":u"Монгол хэл",	"na":u"Ekakairũ Naoero",	"nv":u"Diné bizaad",	"nd":u"isiNdebele",	"ne":u"नेपाली",	"ng":u"Owambo",	"nb":u"Norsk bokmål",	"nn":u"Norsk nynorsk",	"no":u"Norsk",	"ii":u"ꆈꌠ꒿ Nuosuhxop",	"nr":u"isiNdebele",	"oc":u"occitan, lenga d'òc",	"oj":u"ᐊᓂᔑᓈᐯᒧᐎᓐ",	"cu":u"ѩзыкъ словѣньскъ",	"om":u"Afaan Oromoo",	"or":u"ଓଡ଼ିଆ",	"os":u"ирон æвзаг",	"pa":u"ਪੰਜਾਬੀ, پنجابی‎",	"pi":u"पाऴि",	"fa":u"فارسی",	"pl":u"język polski, polszczyzna",	"ps":u"پښتو",	"pt":u"português",	"qu":u"Runa Simi, Kichwa",	"rm":u"rumantsch grischun",	"rn":u"Ikirundi",	"ro":u"limba română",	"rh":u"荣同话",	"ru":u"Русский",	"sa":u"संस्कृतम्",	"sc":u"sardu",	"sd":u"सिन्धी, سنڌي، سندھی‎",	"se":u"Davvisámegiella",	"sm":u"gagana fa'a Samoa",	"sg":u"yângâ tî sängö",	"sr":u"српски језик",	"gd":u"Gàidhlig",	"sn":u"chiShona",	"si":u"සිංහල",	"sk":u"slovenčina, slovenský jazyk",	"sl":u"slovenski jezik, slovenščina",	"so":u"Soomaaliga, af Soomaali",	"st":u"Sesotho",	"es":u"español",	"su":u"Basa Sunda",	"sw":u"Kiswahili",	"ss":u"SiSwati",	"sv":u"svenska",	"ta":u"தமிழ்",	"te":u"తెలుగు",	"tg":u"тоҷикӣ, toçikī, تاجیکی‎",	"th":u"ไทย",	"ti":u"ትግርኛ",	"bo":u"བོད་ཡིག",	"tk":u"Türkmen, Түркмен",	"tl":u"Wikang Tagalog, ᜏᜒᜃᜅ᜔ ᜆᜄᜎᜓᜄ᜔",	"tn":u"Setswana",	"to":u"faka Tonga",	"tr":u"Türkçe",	"ts":u"Xitsonga",	"tt":u"татар теле, tatar tele",	"tw":u"Twi",	"ty":u"Reo Tahiti",	"ug":u"ئۇيغۇرچە‎, Uyghurche",	"uk":u"українська мова",	"ur":u"اردو",	"uz":u"Oʻzbek, Ўзбек, أۇزبېك‎",	"ve":u"Tshivenḓa",	"vi":u"Tiếng Việt",	"vo":u"Volapük",	"wa":u"walon",	"cy":u"Cymraeg",	"wo":u"Wollof",	"fy":u"Frysk",	"xh":u"isiXhosa",	"yi":u"ייִדיש",	"yo":u"Yorùbá",	"za":u"Saɯ cueŋƅ, Saw cuengh",	"zu":u"isiZulu",}



#CONSTANTS FOR WIKIPEDIA READING
BUFFER_SIZE=30000
TEXT_PATTERN=re.compile("<text .*>(.+?)</text>",re.DOTALL)
TITLE_PATTERN=re.compile("<title>(.+?)</title>")
DEBUG_RE=False

REPLACE_PATTERNS=(
    (re.compile("\&quot\;"),'"'),   # &quot; by "
    (re.compile("\&amp\;"),'&'),    # &amp; by &
    (re.compile("\&nbsp\;")," "),       # &nbsp\ by "space"

    (re.compile("&lt;"),'<'),    # &lt; by <
    (re.compile("&gt;"),'>'),    # &lt; by >
    
    (re.compile("'''"),'"'),        # ''' by "
    (re.compile("''"),'"'),         # '' by "

    (re.compile("\«"),'"'),         # « by "
    (re.compile("\»"),'"'),         # » by "

    (re.compile("\<[a-z]+? [^/>]*?\/\>"),""),         # remove short tags <ref name="EB1910" />
    (re.compile("\<(?P<tag>[a-z]+?)\>.*?\<\/(?P=tag)\>",re.DOTALL),""),  # remove all other tags
    (re.compile("\<(?P<tag>[a-z]+?) [^/>]*?\>.+?\<\/(?P=tag)\>",re.DOTALL),""),  # remove all other tags

    (re.compile("\<\!(?P<tag>\-+?).+?(?P=tag)\>",re.DOTALL),""),    #remove comments in <!-- ... -->
    )

CLEANING_PATTERNS=(
    re.compile("\{\{.+?\}\}"),              # first removes everything in double backets in a line
    re.compile("\{\{.+?\}\}",re.DOTALL),    # removes everything else in double backets

    re.compile("\{\|.+?\|\}",re.DOTALL),     # removes everything in {|   |}

    re.compile("\[\[([^\:\|\]\[]+?)\]\]"),#,re.UNICODE),  #simple wikilinks  [[political philosophy]]
    re.compile("\[\[[^\:\|\]\[]+?\|([^\:\|\]\[]+?)\]\]"),#,re.UNICODE),  #simple wikilinks  [[social anarchism|social]]
    re.compile("\[\[.+?\|([^\|]+?)\]\]"),# [[File:Jarach and Zerzan.JPG|thumb|left|Lawrence Jarach (left) and John Zerzan (right), two prominent contemporary anarchist authors. Zerzan is known as prominent voice within anarcho-primitivism, while Jarach is a noted advocate of post-left anarchy.]]
    re.compile("\[\[([^\|\]\[]+?)\]\]"),#,re.UNICODE),  #simple wikilinks with ":" "[[Philosophy: Who Needs It]]"
    
    re.compile("\[htt.+? (.+?)\]"),# external links  [http://www.tvu.com/metalreflectivityLR.jpg reflectivity of metals (chart)]
    re.compile("\[htt.+?\]"),# external links  [http://www.tvu.com/metalreflectivityLR.jpg]
    
    re.compile("^\=+",re.MULTILINE),     # remove initial title, subtitle, ... marks
    re.compile("\=+?$",re.MULTILINE),    # remove final title, subtitle, ... marks
    re.compile("^[\*]+",re.MULTILINE),    # remove initial "bullets"

    # final cleaning
    re.compile("^ ",re.MULTILINE),  # remove spaces at the begining of a line
    re.compile("^ ",re.MULTILINE),
    re.compile(" $",re.MULTILINE),  # remove spaces at the end of a line
    re.compile(" $",re.MULTILINE),
    re.compile(" $",re.MULTILINE),  # remove spaces at the end of a line
    
    re.compile("\n(\n\n)"),  # remove double blank lines
    re.compile("\n(\n\n)"),
    re.compile("\n(\n\n)"),
    re.compile("\n(\n\n)"),
    re.compile("\n(\n\n)"),
    re.compile("\n(\n\n)"),
    re.compile("\(\)"),      # remove ()
    re.compile('"(")'),      # remove resulting double "

    re.compile("^\#",re.MULTILINE), # remove heading # in a line
    re.compile("^\:",re.MULTILINE), # remove heading : in a line
    re.compile("^\:",re.MULTILINE), # remove heading : in a line
    re.compile("^\:",re.MULTILINE), # remove heading : in a line
    
    re.compile("^\;",re.MULTILINE), # remove heading ; in a line
    re.compile("^\.",re.MULTILINE), # remove heading . in a line

    re.compile(" ( )",re.MULTILINE), # remove DOUBLE SPACE
    re.compile(" ( )",re.MULTILINE), # remove DOUBLE SPACE

    re.compile("^\*",re.MULTILINE), # remove heading * in a line
    )


def get_pages(url):
    response = urllib2.urlopen(url)
    decompressor = bz2.BZ2Decompressor()
    data=u""
    #data=""
    
    while True:
        try:
            compressed_data=response.read(BUFFER_SIZE)
        except:
            #print "no more data available in bz2 file"
            break
        try:
            uncompressed_data=decompressor.decompress(compressed_data)
        except:
            #print "ERROR: data not decompressed, more data needed!"
            continue
        
        decoded_data=u""
        try:
            decoded_data+=uncompressed_data.decode("utf-8")
        except:
            pass
            #print "ERROR: decoding data, bye!",len(uncompressed_data)
            #continue
        
        data+=decoded_data
            
        while True:
            pos_start_page=data.find("<page>")
            if pos_start_page==-1:
                break
            pos_end_page=data.find("</page>")
            if pos_end_page==-1:
                break
            if pos_end_page>pos_start_page:
                #print "*",
                yield data[pos_start_page:pos_end_page+7]
                data=data[pos_end_page+7:]
            else:
            	data=data[pos_start_page:]
            	break

def get_articles(url,only_text=True):
    for page in get_pages(url):
        #print page[:10000]
        match_title=TITLE_PATTERN.search(page)
        if match_title!=None:
            title=match_title.group(1)+"\n"
        else:
            title="\n"
        match=TEXT_PATTERN.search(page)
        if match!=None:
            text=match.group(1)
            if text.startswith("#REDIRECT"):
                if not only_text:
                    yield title+text
            else:
                yield title+text

    
def clean_article(article):
    def t(m):
        if DEBUG_RE:
            print
            print "\tREP:",m.group(0)
            print
        return ""
    def t1(m):
        if DEBUG_RE:
            print
            print "\tCLEAN:",m.group(0),"-",m.group(1)
            print
        return m.group(1)
    for replace_pattern,replacement in REPLACE_PATTERNS:
        article=replace_pattern.sub(replacement,article)
    for cleaning_pattern in CLEANING_PATTERNS:
        try:
            article=cleaning_pattern.sub(t1,article)
        except:
            article=cleaning_pattern.sub(t,article)
    re.purge()
    return article
        
    

# http://dumps.wikimedia.org/backup-index.html
WIKIPEDIA_URLS={
#"en":"http://dumps.wikimedia.org/enwiki/20151002/enwiki-20151002-pages-meta-current.xml.bz2",
"en":"https://dumps.wikimedia.org/enwiki/20151102/enwiki-20151102-pages-meta-current.xml.bz2", #ACUALIZADO NOV 2015
"es":"http://dumps.wikimedia.org/eswiki/20151102/eswiki-20151102-pages-meta-current.xml.bz2",
"fr":"http://dumps.wikimedia.org/frwiki/20151020/frwiki-20151020-pages-meta-current1.xml.bz2",
"de":"http://dumps.wikimedia.org/dewiki/20151102/dewiki-20151102-pages-meta-current.xml.bz2",
"it":"http://dumps.wikimedia.org/itwiki/20151102/itwiki-20151102-pages-meta-current.xml.bz2",
"pl":"http://dumps.wikimedia.org/plwiki/20151102/plwiki-20151102-pages-meta-current.xml.bz2",
"nl":"http://dumps.wikimedia.org/nlwiki/20151102/nlwiki-20151102-pages-meta-current.xml.bz2",
"pt":"http://dumps.wikimedia.org/ptwiki/20151102/ptwiki-20151102-pages-meta-current.xml.bz2",
"ru":"http://dumps.wikimedia.org/ruwiki/20151102/ruwiki-20151102-pages-meta-current.xml.bz2",#decode err
"el":"http://dumps.wikimedia.org/elwiki/20151102/elwiki-20151102-pages-meta-current.xml.bz2",#decode err
"ca":"http://dumps.wikimedia.org/cawiki/20151102/cawiki-20151102-pages-meta-current.xml.bz2",
"da":"http://dumps.wikimedia.org/dawiki/20151102/dawiki-20151102-pages-meta-current.xml.bz2",
"ro":"http://dumps.wikimedia.org/rowiki/20151102/rowiki-20151102-pages-meta-current.xml.bz2",
"tr":"http://dumps.wikimedia.org/trwiki/20151102/trwiki-20151102-pages-meta-current.xml.bz2",
#"az":"http://dumps.wikimedia.org/azwiki/20151102/azwiki-20151102-pages-meta-current.xml.bz2",#no
#"bg":"https://dumps.wikimedia.org/bgwiki/20151102/bgwiki-20151102-pages-meta-current.xml.bz2",#decode err
"be":"https://dumps.wikimedia.org/bewiki/20151102/bewiki-20151102-pages-meta-current.xml.bz2",
"cs":"https://dumps.wikimedia.org/cswiki/20151102/cswiki-20151102-pages-articles.xml.bz2",
"et":"https://dumps.wikimedia.org/etwiki/20151102/etwiki-20151102-pages-articles.xml.bz2",
"eo":"https://dumps.wikimedia.org/eowiki/20151102/eowiki-20151102-pages-meta-current.xml.bz2",
"eu":"https://dumps.wikimedia.org/euwiki/20151102/euwiki-20151102-pages-meta-current.xml.bz2",
#"fa":"https://dumps.wikimedia.org/fawiki/20151102/fawiki-20151102-pages-meta-current.xml.bz2",# too many errors ???
"gl":"https://dumps.wikimedia.org/glwiki/20151102/glwiki-20151102-pages-meta-current.xml.bz2",
#"hy":"https://dumps.wikimedia.org/hywiki/20151102/hywiki-20151102-pages-meta-current.xml.bz2",#decode err
#"hi":"https://dumps.wikimedia.org/hiwiki/20151102/hiwiki-20151102-pages-meta-current.xml.bz2",#decode err
"hr":"https://dumps.wikimedia.org/hrwiki/20151102/hrwiki-20151102-pages-meta-current.xml.bz2",
"id":"https://dumps.wikimedia.org/idwiki/20151102/idwiki-20151102-pages-meta-current.xml.bz2",
"he":"https://dumps.wikimedia.org/hewiki/20151102/hewiki-20151102-pages-meta-current.xml.bz2",
#"ka":"https://dumps.wikimedia.org/kawiki/20151102/kawiki-20151102-pages-meta-current.xml.bz2",# very few and too short articles
"la":"https://dumps.wikimedia.org/lawiki/20151102/lawiki-20151102-pages-meta-current.xml.bz2",
"lt":"https://dumps.wikimedia.org/ltwiki/20151102/ltwiki-20151102-pages-meta-current.xml.bz2",
"hu":"https://dumps.wikimedia.org/ltwiki/20151102/ltwiki-20151102-pages-meta-current.xml.bz2",
"ms":"https://dumps.wikimedia.org/mswiki/20151102/mswiki-20151102-pages-meta-current.xml.bz2",
#"ja":"https://dumps.wikimedia.org/jawiki/20151020/jawiki-20151020-pages-meta-current.xml.bz2",#decode err
#"no":"https://dumps.wikimedia.org/nowiki/20151102/nowiki-20151102-pages-meta-current.xml.bz2",#decode err
"ce":"https://dumps.wikimedia.org/cewiki/20151102/cewiki-20151102-pages-meta-current.xml.bz2",
"uz":"https://dumps.wikimedia.org/uzwiki/20151102/uzwiki-20151102-pages-meta-current.xml.bz2",
#"kk":"https://dumps.wikimedia.org/kkwiki/20151102/kkwiki-20151102-pages-meta-current.xml.bz2",#decode err
"sk":"https://dumps.wikimedia.org/skwiki/20151102/skwiki-20151102-pages-meta-current.xml.bz2",
"sl":"https://dumps.wikimedia.org/slwiki/20151102/slwiki-20151102-pages-meta-current.xml.bz2",
"sr":"https://dumps.wikimedia.org/srwiki/20151102/srwiki-20151102-pages-meta-current.xml.bz2",
#"sh":"https://dumps.wikimedia.org/shwiki/20151102/shwiki-20151102-pages-meta-current.xml.bz2",#  very few and too short articles
"fi":"https://dumps.wikimedia.org/fiwiki/20151102/fiwiki-20151102-pages-meta-current.xml.bz2",
"uk":"https://dumps.wikimedia.org/ukwiki/20151102/ukwiki-20151102-pages-meta-current.xml.bz2",
}

