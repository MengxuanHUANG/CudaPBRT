#pragma once

#include <unordered_map>
#ifndef MY_INPUT_CODE
#define MY_INPUT_CODE
// From glfw3.h
#define MY_KEY_SPACE              32
#define MY_KEY_APOSTROPHE         39  /* ' */
#define MY_KEY_COMMA              44  /* , */
#define MY_KEY_MINUS              45  /* - */
#define MY_KEY_PERIOD             46  /* . */
#define MY_KEY_SLASH              47  /* / */
#define MY_KEY_0                  48
#define MY_KEY_1                  49
#define MY_KEY_2                  50
#define MY_KEY_3                  51
#define MY_KEY_4                  52
#define MY_KEY_5                  53
#define MY_KEY_6                  54
#define MY_KEY_7                  55
#define MY_KEY_8                  56
#define MY_KEY_9                  57
#define MY_KEY_SEMICOLON          59  /* ; */
#define MY_KEY_EQUAL              61  /* = */
#define MY_KEY_A                  65
#define MY_KEY_B                  66
#define MY_KEY_C                  67
#define MY_KEY_D                  68
#define MY_KEY_E                  69
#define MY_KEY_F                  70
#define MY_KEY_G                  71
#define MY_KEY_H                  72
#define MY_KEY_I                  73
#define MY_KEY_J                  74
#define MY_KEY_K                  75
#define MY_KEY_L                  76
#define MY_KEY_M                  77
#define MY_KEY_N                  78
#define MY_KEY_O                  79
#define MY_KEY_P                  80
#define MY_KEY_Q                  81
#define MY_KEY_R                  82
#define MY_KEY_S                  83
#define MY_KEY_T                  84
#define MY_KEY_U                  85
#define MY_KEY_V                  86
#define MY_KEY_W                  87
#define MY_KEY_X                  88
#define MY_KEY_Y                  89
#define MY_KEY_Z                  90
#define MY_KEY_LEFT_BRACKET       91  /* [ */
#define MY_KEY_BACKSLASH          92  /* \ */
#define MY_KEY_RIGHT_BRACKET      93  /* ] */
#define MY_KEY_GRAVE_ACCENT       96  /* ` */
#define MY_KEY_WORLD_1            161 /* non-US #1 */
#define MY_KEY_WORLD_2            162 /* non-US #2 */

	/* Function keys */
#define MY_KEY_ESCAPE             256
#define MY_KEY_ENTER              257
#define MY_KEY_TAB                258
#define MY_KEY_BACKSPACE          259
#define MY_KEY_INSERT             260
#define MY_KEY_DELETE             261
#define MY_KEY_RIGHT              262
#define MY_KEY_LEFT               263
#define MY_KEY_DOWN               264
#define MY_KEY_UP                 265
#define MY_KEY_PAGE_UP            266
#define MY_KEY_PAGE_DOWN          267
#define MY_KEY_HOME               268
#define MY_KEY_END                269
#define MY_KEY_CAPS_LOCK          280
#define MY_KEY_SCROLL_LOCK        281
#define MY_KEY_NUM_LOCK           282
#define MY_KEY_PRINT_SCREEN       283
#define MY_KEY_PAUSE              284
#define MY_KEY_F1                 290
#define MY_KEY_F2                 291
#define MY_KEY_F3                 292
#define MY_KEY_F4                 293
#define MY_KEY_F5                 294
#define MY_KEY_F6                 295
#define MY_KEY_F7                 296
#define MY_KEY_F8                 297
#define MY_KEY_F9                 298
#define MY_KEY_F10                299
#define MY_KEY_F11                300
#define MY_KEY_F12                301
#define MY_KEY_F13                302
#define MY_KEY_F14                303
#define MY_KEY_F15                304
#define MY_KEY_F16                305
#define MY_KEY_F17                306
#define MY_KEY_F18                307
#define MY_KEY_F19                308
#define MY_KEY_F20                309
#define MY_KEY_F21                310
#define MY_KEY_F22                311
#define MY_KEY_F23                312
#define MY_KEY_F24                313
#define MY_KEY_F25                314
#define MY_KEY_KP_0               320
#define MY_KEY_KP_1               321
#define MY_KEY_KP_2               322
#define MY_KEY_KP_3               323
#define MY_KEY_KP_4               324
#define MY_KEY_KP_5               325
#define MY_KEY_KP_6               326
#define MY_KEY_KP_7               327
#define MY_KEY_KP_8               328
#define MY_KEY_KP_9               329
#define MY_KEY_KP_DECIMAL         330
#define MY_KEY_KP_DIVIDE          331
#define MY_KEY_KP_MULTIPLY        332
#define MY_KEY_KP_SUBTRACT        333
#define MY_KEY_KP_ADD             334
#define MY_KEY_KP_ENTER           335
#define MY_KEY_KP_EQUAL           336
#define MY_KEY_LEFT_SHIFT         340
#define MY_KEY_LEFT_CONTROL       341
#define MY_KEY_LEFT_ALT           342
#define MY_KEY_LEFT_SUPER         343
#define MY_KEY_RIGHT_SHIFT        344
#define MY_KEY_RIGHT_CONTROL      345
#define MY_KEY_RIGHT_ALT          346
#define MY_KEY_RIGHT_SUPER        347
#define MY_KEY_MENU               348

#define MY_MOUSE_BN_LEFT          0
#define MY_MOUSE_BN_RIGHT         1
#define MY_MOUSE_BN_MIDDLE        2
#define MY_MOUSE_BN_EXTRA_1       3
#define MY_MOUSE_BN_EXTRA_2       4

#endif

namespace CudaPBRT
{
	const std::unordered_map<int, const char*> KeyMap{
		{32,"SPACE"},
		{39 ,"APOSTROPHE"},
		{44 ,"COMMA"},
		{45 ,"MINUS"},
		{46 ,"PERIOD"},
		{47 ,"SLASH"},
		{48,"NUM_0"},
		{49,"NUM_1"},
		{50,"NUM_2"},
		{51,"NUM_3"},
		{52,"NUM_4"},
		{53,"NUM_5"},
		{54,"NUM_6"},
		{55,"NUM_7"},
		{56,"NUM_8"},
		{57,"NUM_9"},
		{59 ,"SEMICOLON"},
		{61 ,"EQUAL"},
		{65,"A"},
		{66,"B"},
		{67,"C"},
		{68,"D"},
		{69,"E"},
		{70,"F"},
		{71,"G"},
		{72,"H"},
		{73,"I"},
		{74,"J"},
		{75,"K"},
		{76,"L"},
		{77,"M"},
		{78,"N"},
		{79,"O"},
		{80,"P"},
		{81,"Q"},
		{82,"R"},
		{83,"S"},
		{84,"T"},
		{85,"U"},
		{86,"V"},
		{87,"W"},
		{88,"X"},
		{89,"Y"},
		{90,"Z"},
		{91 ,"LEFT_BRACKET"},
		{92 ,"BACKSLASH"},
		{93 ,"RIGHT_BRACKET"},
		{96 ,"GRAVE_ACCENT"},
		{161,"WORLD_1"},
		{162,"WORLD_2"},
		{256, "ESCAPE"},
		{257, "ENTER"},
		{258, "TAB"},
		{259, "BACKSPACE"},
		{260, "INSERT"},
		{261, "DELETE"},
		{262, "RIGHT"},
		{263, "LEFT"},
		{264, "DOWN"},
		{265, "UP"},
		{266, "PAGE_UP"},
		{267, "PAGE_DOWN"},
		{268, "HOME"},
		{269, "END"},
		{280, "CAPS_LOCK"},
		{281, "SCROLL_LOCK"},
		{282, "NUM_LOCK"},
		{283, "PRINT_SCREEN"},
		{284, "PAUSE"},
		{290, "F1"},
		{291, "F2"},
		{292, "F3"},
		{293, "F4"},
		{294, "F5"},
		{295, "F6"},
		{296, "F7"},
		{297, "F8"},
		{298, "F9"},
		{299, "F10"},
		{300, "F11"},
		{301, "F12"},
		{302, "F13"},
		{303, "F14"},
		{304, "F15"},
		{305, "F16"},
		{306, "F17"},
		{307, "F18"},
		{308, "F19"},
		{309, "F20"},
		{310, "F21"},
		{311, "F22"},
		{312, "F23"},
		{313, "F24"},
		{314, "F25"},
		{320, "KP_0"},
		{321, "KP_1"},
		{322, "KP_2"},
		{323, "KP_3"},
		{324, "KP_4"},
		{325, "KP_5"},
		{326, "KP_6"},
		{327, "KP_7"},
		{328, "KP_8"},
		{329, "KP_9"},
		{330, "KP_DECIMAL"},
		{331, "KP_DIVIDE"},
		{332, "KP_MULTIPLY"},
		{333, "KP_SUBTRACT"},
		{334, "KP_ADD"},
		{335, "KP_ENTER"},
		{336, "KP_EQUAL"},
		{340, "LEFT_SHIFT"},
		{341, "LEFT_CONTROL"},
		{342, "LEFT_ALT"},
		{343, "LEFT_SUPER"},
		{344, "RIGHT_SHIFT"},
		{345, "RIGHT_CONTROL"},
		{346, "RIGHT_ALT"},
		{347, "RIGHT_SUPER"},
		{348, "MENU"},
	};
}
