

// SFW - Simple Framework - https://github.com/Relintai/sfw

// Simple c++ app / game framework inspired by the single file c game engines
// and libraries, especially [FWK](https://github.com/r-lyeh/FWK).

// MODULES:
// CORE + OBJECT

// LICENSES:

/*
SFW - (MIT)

Copyright (c) 2023-present Péter Magyar.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// This Library uses code from the following projects:

// Pandemonium Engine (MIT)

/*************************************************************************/
/*                     PANDEMONIUM ENGINE LICENSE                        */
/*             https://github.com/Relintai/pandemonium_engine            */
/*************************************************************************/
/* Copyright (c) 2022-present Péter Magyar.                              */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

// FWK

//https://github.com/r-lyeh/FWK

/*

Unlicense

This software is released into the [public domain](https://unlicense.org/).
Also dual-licensed as [0-BSD](https://opensource.org/licenses/0BSD) or [MIT (No Attribution)](https://github.com/aws/mit-0)
for those countries where public domain is a concern (sigh).
Any contribution to this repository is implicitly subjected to the same release conditions aforementioned.

*/

// Trantor

//inet_address.h, .cpp, socket.h and .cpp originates from trantor (they were heavily modified):

// Trantor - A non-blocking I/O based TCP network library, using C++14/17,
// Copyright (c) 2016-2021, Tao An.  All rights reserved.
// https://github.com/an-tao/trantor
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//   * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//   * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//   * Neither the name of Tao An nor the names of other contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Muduo - A reactor-based C++ network library for Linux
// Copyright (c) 2010, Shuo Chen.  All rights reserved.
// http://code.google.com/p/muduo/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//   * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//   * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//   * Neither the name of Shuo Chen nor the names of other contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// PCG

// pcg.h's License:

// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)

#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <wchar.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>

#if !defined(_WIN64) && !defined(_WIN32)
#include <sys/time.h>
#include <unistd.h>
#endif

#ifndef SFW_H
#include "sfw.h"
#endif

//===================  CORE SECTION  ===================

#line 1 "sfw/core/ucaps.h"

/*************************************************************************/
/*  ucaps.h                                                              */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

//satan invented unicode?
#define CAPS_LEN 666

static const int caps_table[CAPS_LEN][2] = {
	{ 0x0061, 0x0041 },
	{ 0x0062, 0x0042 },
	{ 0x0063, 0x0043 },
	{ 0x0064, 0x0044 },
	{ 0x0065, 0x0045 },
	{ 0x0066, 0x0046 },
	{ 0x0067, 0x0047 },
	{ 0x0068, 0x0048 },
	{ 0x0069, 0x0049 },
	{ 0x006A, 0x004A },
	{ 0x006B, 0x004B },
	{ 0x006C, 0x004C },
	{ 0x006D, 0x004D },
	{ 0x006E, 0x004E },
	{ 0x006F, 0x004F },
	{ 0x0070, 0x0050 },
	{ 0x0071, 0x0051 },
	{ 0x0072, 0x0052 },
	{ 0x0073, 0x0053 },
	{ 0x0074, 0x0054 },
	{ 0x0075, 0x0055 },
	{ 0x0076, 0x0056 },
	{ 0x0077, 0x0057 },
	{ 0x0078, 0x0058 },
	{ 0x0079, 0x0059 },
	{ 0x007A, 0x005A },
	{ 0x00E0, 0x00C0 },
	{ 0x00E1, 0x00C1 },
	{ 0x00E2, 0x00C2 },
	{ 0x00E3, 0x00C3 },
	{ 0x00E4, 0x00C4 },
	{ 0x00E5, 0x00C5 },
	{ 0x00E6, 0x00C6 },
	{ 0x00E7, 0x00C7 },
	{ 0x00E8, 0x00C8 },
	{ 0x00E9, 0x00C9 },
	{ 0x00EA, 0x00CA },
	{ 0x00EB, 0x00CB },
	{ 0x00EC, 0x00CC },
	{ 0x00ED, 0x00CD },
	{ 0x00EE, 0x00CE },
	{ 0x00EF, 0x00CF },
	{ 0x00F0, 0x00D0 },
	{ 0x00F1, 0x00D1 },
	{ 0x00F2, 0x00D2 },
	{ 0x00F3, 0x00D3 },
	{ 0x00F4, 0x00D4 },
	{ 0x00F5, 0x00D5 },
	{ 0x00F6, 0x00D6 },
	{ 0x00F8, 0x00D8 },
	{ 0x00F9, 0x00D9 },
	{ 0x00FA, 0x00DA },
	{ 0x00FB, 0x00DB },
	{ 0x00FC, 0x00DC },
	{ 0x00FD, 0x00DD },
	{ 0x00FE, 0x00DE },
	{ 0x00FF, 0x0178 },
	{ 0x0101, 0x0100 },
	{ 0x0103, 0x0102 },
	{ 0x0105, 0x0104 },
	{ 0x0107, 0x0106 },
	{ 0x0109, 0x0108 },
	{ 0x010B, 0x010A },
	{ 0x010D, 0x010C },
	{ 0x010F, 0x010E },
	{ 0x0111, 0x0110 },
	{ 0x0113, 0x0112 },
	{ 0x0115, 0x0114 },
	{ 0x0117, 0x0116 },
	{ 0x0119, 0x0118 },
	{ 0x011B, 0x011A },
	{ 0x011D, 0x011C },
	{ 0x011F, 0x011E },
	{ 0x0121, 0x0120 },
	{ 0x0123, 0x0122 },
	{ 0x0125, 0x0124 },
	{ 0x0127, 0x0126 },
	{ 0x0129, 0x0128 },
	{ 0x012B, 0x012A },
	{ 0x012D, 0x012C },
	{ 0x012F, 0x012E },
	{ 0x0131, 0x0049 },
	{ 0x0133, 0x0132 },
	{ 0x0135, 0x0134 },
	{ 0x0137, 0x0136 },
	{ 0x013A, 0x0139 },
	{ 0x013C, 0x013B },
	{ 0x013E, 0x013D },
	{ 0x0140, 0x013F },
	{ 0x0142, 0x0141 },
	{ 0x0144, 0x0143 },
	{ 0x0146, 0x0145 },
	{ 0x0148, 0x0147 },
	{ 0x014B, 0x014A },
	{ 0x014D, 0x014C },
	{ 0x014F, 0x014E },
	{ 0x0151, 0x0150 },
	{ 0x0153, 0x0152 },
	{ 0x0155, 0x0154 },
	{ 0x0157, 0x0156 },
	{ 0x0159, 0x0158 },
	{ 0x015B, 0x015A },
	{ 0x015D, 0x015C },
	{ 0x015F, 0x015E },
	{ 0x0161, 0x0160 },
	{ 0x0163, 0x0162 },
	{ 0x0165, 0x0164 },
	{ 0x0167, 0x0166 },
	{ 0x0169, 0x0168 },
	{ 0x016B, 0x016A },
	{ 0x016D, 0x016C },
	{ 0x016F, 0x016E },
	{ 0x0171, 0x0170 },
	{ 0x0173, 0x0172 },
	{ 0x0175, 0x0174 },
	{ 0x0177, 0x0176 },
	{ 0x017A, 0x0179 },
	{ 0x017C, 0x017B },
	{ 0x017E, 0x017D },
	{ 0x0183, 0x0182 },
	{ 0x0185, 0x0184 },
	{ 0x0188, 0x0187 },
	{ 0x018C, 0x018B },
	{ 0x0192, 0x0191 },
	{ 0x0199, 0x0198 },
	{ 0x01A1, 0x01A0 },
	{ 0x01A3, 0x01A2 },
	{ 0x01A5, 0x01A4 },
	{ 0x01A8, 0x01A7 },
	{ 0x01AD, 0x01AC },
	{ 0x01B0, 0x01AF },
	{ 0x01B4, 0x01B3 },
	{ 0x01B6, 0x01B5 },
	{ 0x01B9, 0x01B8 },
	{ 0x01BD, 0x01BC },
	{ 0x01C6, 0x01C4 },
	{ 0x01C9, 0x01C7 },
	{ 0x01CC, 0x01CA },
	{ 0x01CE, 0x01CD },
	{ 0x01D0, 0x01CF },
	{ 0x01D2, 0x01D1 },
	{ 0x01D4, 0x01D3 },
	{ 0x01D6, 0x01D5 },
	{ 0x01D8, 0x01D7 },
	{ 0x01DA, 0x01D9 },
	{ 0x01DC, 0x01DB },
	{ 0x01DF, 0x01DE },
	{ 0x01E1, 0x01E0 },
	{ 0x01E3, 0x01E2 },
	{ 0x01E5, 0x01E4 },
	{ 0x01E7, 0x01E6 },
	{ 0x01E9, 0x01E8 },
	{ 0x01EB, 0x01EA },
	{ 0x01ED, 0x01EC },
	{ 0x01EF, 0x01EE },
	{ 0x01F3, 0x01F1 },
	{ 0x01F5, 0x01F4 },
	{ 0x01FB, 0x01FA },
	{ 0x01FD, 0x01FC },
	{ 0x01FF, 0x01FE },
	{ 0x0201, 0x0200 },
	{ 0x0203, 0x0202 },
	{ 0x0205, 0x0204 },
	{ 0x0207, 0x0206 },
	{ 0x0209, 0x0208 },
	{ 0x020B, 0x020A },
	{ 0x020D, 0x020C },
	{ 0x020F, 0x020E },
	{ 0x0211, 0x0210 },
	{ 0x0213, 0x0212 },
	{ 0x0215, 0x0214 },
	{ 0x0217, 0x0216 },
	{ 0x0253, 0x0181 },
	{ 0x0254, 0x0186 },
	{ 0x0257, 0x018A },
	{ 0x0258, 0x018E },
	{ 0x0259, 0x018F },
	{ 0x025B, 0x0190 },
	{ 0x0260, 0x0193 },
	{ 0x0263, 0x0194 },
	{ 0x0268, 0x0197 },
	{ 0x0269, 0x0196 },
	{ 0x026F, 0x019C },
	{ 0x0272, 0x019D },
	{ 0x0275, 0x019F },
	{ 0x0283, 0x01A9 },
	{ 0x0288, 0x01AE },
	{ 0x028A, 0x01B1 },
	{ 0x028B, 0x01B2 },
	{ 0x0292, 0x01B7 },
	{ 0x03AC, 0x0386 },
	{ 0x03AD, 0x0388 },
	{ 0x03AE, 0x0389 },
	{ 0x03AF, 0x038A },
	{ 0x03B1, 0x0391 },
	{ 0x03B2, 0x0392 },
	{ 0x03B3, 0x0393 },
	{ 0x03B4, 0x0394 },
	{ 0x03B5, 0x0395 },
	{ 0x03B6, 0x0396 },
	{ 0x03B7, 0x0397 },
	{ 0x03B8, 0x0398 },
	{ 0x03B9, 0x0399 },
	{ 0x03BA, 0x039A },
	{ 0x03BB, 0x039B },
	{ 0x03BC, 0x039C },
	{ 0x03BD, 0x039D },
	{ 0x03BE, 0x039E },
	{ 0x03BF, 0x039F },
	{ 0x03C0, 0x03A0 },
	{ 0x03C1, 0x03A1 },
	{ 0x03C3, 0x03A3 },
	{ 0x03C4, 0x03A4 },
	{ 0x03C5, 0x03A5 },
	{ 0x03C6, 0x03A6 },
	{ 0x03C7, 0x03A7 },
	{ 0x03C8, 0x03A8 },
	{ 0x03C9, 0x03A9 },
	{ 0x03CA, 0x03AA },
	{ 0x03CB, 0x03AB },
	{ 0x03CC, 0x038C },
	{ 0x03CD, 0x038E },
	{ 0x03CE, 0x038F },
	{ 0x03E3, 0x03E2 },
	{ 0x03E5, 0x03E4 },
	{ 0x03E7, 0x03E6 },
	{ 0x03E9, 0x03E8 },
	{ 0x03EB, 0x03EA },
	{ 0x03ED, 0x03EC },
	{ 0x03EF, 0x03EE },
	{ 0x0430, 0x0410 },
	{ 0x0431, 0x0411 },
	{ 0x0432, 0x0412 },
	{ 0x0433, 0x0413 },
	{ 0x0434, 0x0414 },
	{ 0x0435, 0x0415 },
	{ 0x0436, 0x0416 },
	{ 0x0437, 0x0417 },
	{ 0x0438, 0x0418 },
	{ 0x0439, 0x0419 },
	{ 0x043A, 0x041A },
	{ 0x043B, 0x041B },
	{ 0x043C, 0x041C },
	{ 0x043D, 0x041D },
	{ 0x043E, 0x041E },
	{ 0x043F, 0x041F },
	{ 0x0440, 0x0420 },
	{ 0x0441, 0x0421 },
	{ 0x0442, 0x0422 },
	{ 0x0443, 0x0423 },
	{ 0x0444, 0x0424 },
	{ 0x0445, 0x0425 },
	{ 0x0446, 0x0426 },
	{ 0x0447, 0x0427 },
	{ 0x0448, 0x0428 },
	{ 0x0449, 0x0429 },
	{ 0x044A, 0x042A },
	{ 0x044B, 0x042B },
	{ 0x044C, 0x042C },
	{ 0x044D, 0x042D },
	{ 0x044E, 0x042E },
	{ 0x044F, 0x042F },
	{ 0x0451, 0x0401 },
	{ 0x0452, 0x0402 },
	{ 0x0453, 0x0403 },
	{ 0x0454, 0x0404 },
	{ 0x0455, 0x0405 },
	{ 0x0456, 0x0406 },
	{ 0x0457, 0x0407 },
	{ 0x0458, 0x0408 },
	{ 0x0459, 0x0409 },
	{ 0x045A, 0x040A },
	{ 0x045B, 0x040B },
	{ 0x045C, 0x040C },
	{ 0x045E, 0x040E },
	{ 0x045F, 0x040F },
	{ 0x0461, 0x0460 },
	{ 0x0463, 0x0462 },
	{ 0x0465, 0x0464 },
	{ 0x0467, 0x0466 },
	{ 0x0469, 0x0468 },
	{ 0x046B, 0x046A },
	{ 0x046D, 0x046C },
	{ 0x046F, 0x046E },
	{ 0x0471, 0x0470 },
	{ 0x0473, 0x0472 },
	{ 0x0475, 0x0474 },
	{ 0x0477, 0x0476 },
	{ 0x0479, 0x0478 },
	{ 0x047B, 0x047A },
	{ 0x047D, 0x047C },
	{ 0x047F, 0x047E },
	{ 0x0481, 0x0480 },
	{ 0x0491, 0x0490 },
	{ 0x0493, 0x0492 },
	{ 0x0495, 0x0494 },
	{ 0x0497, 0x0496 },
	{ 0x0499, 0x0498 },
	{ 0x049B, 0x049A },
	{ 0x049D, 0x049C },
	{ 0x049F, 0x049E },
	{ 0x04A1, 0x04A0 },
	{ 0x04A3, 0x04A2 },
	{ 0x04A5, 0x04A4 },
	{ 0x04A7, 0x04A6 },
	{ 0x04A9, 0x04A8 },
	{ 0x04AB, 0x04AA },
	{ 0x04AD, 0x04AC },
	{ 0x04AF, 0x04AE },
	{ 0x04B1, 0x04B0 },
	{ 0x04B3, 0x04B2 },
	{ 0x04B5, 0x04B4 },
	{ 0x04B7, 0x04B6 },
	{ 0x04B9, 0x04B8 },
	{ 0x04BB, 0x04BA },
	{ 0x04BD, 0x04BC },
	{ 0x04BF, 0x04BE },
	{ 0x04C2, 0x04C1 },
	{ 0x04C4, 0x04C3 },
	{ 0x04C8, 0x04C7 },
	{ 0x04CC, 0x04CB },
	{ 0x04D1, 0x04D0 },
	{ 0x04D3, 0x04D2 },
	{ 0x04D5, 0x04D4 },
	{ 0x04D7, 0x04D6 },
	{ 0x04D9, 0x04D8 },
	{ 0x04DB, 0x04DA },
	{ 0x04DD, 0x04DC },
	{ 0x04DF, 0x04DE },
	{ 0x04E1, 0x04E0 },
	{ 0x04E3, 0x04E2 },
	{ 0x04E5, 0x04E4 },
	{ 0x04E7, 0x04E6 },
	{ 0x04E9, 0x04E8 },
	{ 0x04EB, 0x04EA },
	{ 0x04EF, 0x04EE },
	{ 0x04F1, 0x04F0 },
	{ 0x04F3, 0x04F2 },
	{ 0x04F5, 0x04F4 },
	{ 0x04F9, 0x04F8 },
	{ 0x0561, 0x0531 },
	{ 0x0562, 0x0532 },
	{ 0x0563, 0x0533 },
	{ 0x0564, 0x0534 },
	{ 0x0565, 0x0535 },
	{ 0x0566, 0x0536 },
	{ 0x0567, 0x0537 },
	{ 0x0568, 0x0538 },
	{ 0x0569, 0x0539 },
	{ 0x056A, 0x053A },
	{ 0x056B, 0x053B },
	{ 0x056C, 0x053C },
	{ 0x056D, 0x053D },
	{ 0x056E, 0x053E },
	{ 0x056F, 0x053F },
	{ 0x0570, 0x0540 },
	{ 0x0571, 0x0541 },
	{ 0x0572, 0x0542 },
	{ 0x0573, 0x0543 },
	{ 0x0574, 0x0544 },
	{ 0x0575, 0x0545 },
	{ 0x0576, 0x0546 },
	{ 0x0577, 0x0547 },
	{ 0x0578, 0x0548 },
	{ 0x0579, 0x0549 },
	{ 0x057A, 0x054A },
	{ 0x057B, 0x054B },
	{ 0x057C, 0x054C },
	{ 0x057D, 0x054D },
	{ 0x057E, 0x054E },
	{ 0x057F, 0x054F },
	{ 0x0580, 0x0550 },
	{ 0x0581, 0x0551 },
	{ 0x0582, 0x0552 },
	{ 0x0583, 0x0553 },
	{ 0x0584, 0x0554 },
	{ 0x0585, 0x0555 },
	{ 0x0586, 0x0556 },
	{ 0x10D0, 0x10A0 },
	{ 0x10D1, 0x10A1 },
	{ 0x10D2, 0x10A2 },
	{ 0x10D3, 0x10A3 },
	{ 0x10D4, 0x10A4 },
	{ 0x10D5, 0x10A5 },
	{ 0x10D6, 0x10A6 },
	{ 0x10D7, 0x10A7 },
	{ 0x10D8, 0x10A8 },
	{ 0x10D9, 0x10A9 },
	{ 0x10DA, 0x10AA },
	{ 0x10DB, 0x10AB },
	{ 0x10DC, 0x10AC },
	{ 0x10DD, 0x10AD },
	{ 0x10DE, 0x10AE },
	{ 0x10DF, 0x10AF },
	{ 0x10E0, 0x10B0 },
	{ 0x10E1, 0x10B1 },
	{ 0x10E2, 0x10B2 },
	{ 0x10E3, 0x10B3 },
	{ 0x10E4, 0x10B4 },
	{ 0x10E5, 0x10B5 },
	{ 0x10E6, 0x10B6 },
	{ 0x10E7, 0x10B7 },
	{ 0x10E8, 0x10B8 },
	{ 0x10E9, 0x10B9 },
	{ 0x10EA, 0x10BA },
	{ 0x10EB, 0x10BB },
	{ 0x10EC, 0x10BC },
	{ 0x10ED, 0x10BD },
	{ 0x10EE, 0x10BE },
	{ 0x10EF, 0x10BF },
	{ 0x10F0, 0x10C0 },
	{ 0x10F1, 0x10C1 },
	{ 0x10F2, 0x10C2 },
	{ 0x10F3, 0x10C3 },
	{ 0x10F4, 0x10C4 },
	{ 0x10F5, 0x10C5 },
	{ 0x1E01, 0x1E00 },
	{ 0x1E03, 0x1E02 },
	{ 0x1E05, 0x1E04 },
	{ 0x1E07, 0x1E06 },
	{ 0x1E09, 0x1E08 },
	{ 0x1E0B, 0x1E0A },
	{ 0x1E0D, 0x1E0C },
	{ 0x1E0F, 0x1E0E },
	{ 0x1E11, 0x1E10 },
	{ 0x1E13, 0x1E12 },
	{ 0x1E15, 0x1E14 },
	{ 0x1E17, 0x1E16 },
	{ 0x1E19, 0x1E18 },
	{ 0x1E1B, 0x1E1A },
	{ 0x1E1D, 0x1E1C },
	{ 0x1E1F, 0x1E1E },
	{ 0x1E21, 0x1E20 },
	{ 0x1E23, 0x1E22 },
	{ 0x1E25, 0x1E24 },
	{ 0x1E27, 0x1E26 },
	{ 0x1E29, 0x1E28 },
	{ 0x1E2B, 0x1E2A },
	{ 0x1E2D, 0x1E2C },
	{ 0x1E2F, 0x1E2E },
	{ 0x1E31, 0x1E30 },
	{ 0x1E33, 0x1E32 },
	{ 0x1E35, 0x1E34 },
	{ 0x1E37, 0x1E36 },
	{ 0x1E39, 0x1E38 },
	{ 0x1E3B, 0x1E3A },
	{ 0x1E3D, 0x1E3C },
	{ 0x1E3F, 0x1E3E },
	{ 0x1E41, 0x1E40 },
	{ 0x1E43, 0x1E42 },
	{ 0x1E45, 0x1E44 },
	{ 0x1E47, 0x1E46 },
	{ 0x1E49, 0x1E48 },
	{ 0x1E4B, 0x1E4A },
	{ 0x1E4D, 0x1E4C },
	{ 0x1E4F, 0x1E4E },
	{ 0x1E51, 0x1E50 },
	{ 0x1E53, 0x1E52 },
	{ 0x1E55, 0x1E54 },
	{ 0x1E57, 0x1E56 },
	{ 0x1E59, 0x1E58 },
	{ 0x1E5B, 0x1E5A },
	{ 0x1E5D, 0x1E5C },
	{ 0x1E5F, 0x1E5E },
	{ 0x1E61, 0x1E60 },
	{ 0x1E63, 0x1E62 },
	{ 0x1E65, 0x1E64 },
	{ 0x1E67, 0x1E66 },
	{ 0x1E69, 0x1E68 },
	{ 0x1E6B, 0x1E6A },
	{ 0x1E6D, 0x1E6C },
	{ 0x1E6F, 0x1E6E },
	{ 0x1E71, 0x1E70 },
	{ 0x1E73, 0x1E72 },
	{ 0x1E75, 0x1E74 },
	{ 0x1E77, 0x1E76 },
	{ 0x1E79, 0x1E78 },
	{ 0x1E7B, 0x1E7A },
	{ 0x1E7D, 0x1E7C },
	{ 0x1E7F, 0x1E7E },
	{ 0x1E81, 0x1E80 },
	{ 0x1E83, 0x1E82 },
	{ 0x1E85, 0x1E84 },
	{ 0x1E87, 0x1E86 },
	{ 0x1E89, 0x1E88 },
	{ 0x1E8B, 0x1E8A },
	{ 0x1E8D, 0x1E8C },
	{ 0x1E8F, 0x1E8E },
	{ 0x1E91, 0x1E90 },
	{ 0x1E93, 0x1E92 },
	{ 0x1E95, 0x1E94 },
	{ 0x1EA1, 0x1EA0 },
	{ 0x1EA3, 0x1EA2 },
	{ 0x1EA5, 0x1EA4 },
	{ 0x1EA7, 0x1EA6 },
	{ 0x1EA9, 0x1EA8 },
	{ 0x1EAB, 0x1EAA },
	{ 0x1EAD, 0x1EAC },
	{ 0x1EAF, 0x1EAE },
	{ 0x1EB1, 0x1EB0 },
	{ 0x1EB3, 0x1EB2 },
	{ 0x1EB5, 0x1EB4 },
	{ 0x1EB7, 0x1EB6 },
	{ 0x1EB9, 0x1EB8 },
	{ 0x1EBB, 0x1EBA },
	{ 0x1EBD, 0x1EBC },
	{ 0x1EBF, 0x1EBE },
	{ 0x1EC1, 0x1EC0 },
	{ 0x1EC3, 0x1EC2 },
	{ 0x1EC5, 0x1EC4 },
	{ 0x1EC7, 0x1EC6 },
	{ 0x1EC9, 0x1EC8 },
	{ 0x1ECB, 0x1ECA },
	{ 0x1ECD, 0x1ECC },
	{ 0x1ECF, 0x1ECE },
	{ 0x1ED1, 0x1ED0 },
	{ 0x1ED3, 0x1ED2 },
	{ 0x1ED5, 0x1ED4 },
	{ 0x1ED7, 0x1ED6 },
	{ 0x1ED9, 0x1ED8 },
	{ 0x1EDB, 0x1EDA },
	{ 0x1EDD, 0x1EDC },
	{ 0x1EDF, 0x1EDE },
	{ 0x1EE1, 0x1EE0 },
	{ 0x1EE3, 0x1EE2 },
	{ 0x1EE5, 0x1EE4 },
	{ 0x1EE7, 0x1EE6 },
	{ 0x1EE9, 0x1EE8 },
	{ 0x1EEB, 0x1EEA },
	{ 0x1EED, 0x1EEC },
	{ 0x1EEF, 0x1EEE },
	{ 0x1EF1, 0x1EF0 },
	{ 0x1EF3, 0x1EF2 },
	{ 0x1EF5, 0x1EF4 },
	{ 0x1EF7, 0x1EF6 },
	{ 0x1EF9, 0x1EF8 },
	{ 0x1F00, 0x1F08 },
	{ 0x1F01, 0x1F09 },
	{ 0x1F02, 0x1F0A },
	{ 0x1F03, 0x1F0B },
	{ 0x1F04, 0x1F0C },
	{ 0x1F05, 0x1F0D },
	{ 0x1F06, 0x1F0E },
	{ 0x1F07, 0x1F0F },
	{ 0x1F10, 0x1F18 },
	{ 0x1F11, 0x1F19 },
	{ 0x1F12, 0x1F1A },
	{ 0x1F13, 0x1F1B },
	{ 0x1F14, 0x1F1C },
	{ 0x1F15, 0x1F1D },
	{ 0x1F20, 0x1F28 },
	{ 0x1F21, 0x1F29 },
	{ 0x1F22, 0x1F2A },
	{ 0x1F23, 0x1F2B },
	{ 0x1F24, 0x1F2C },
	{ 0x1F25, 0x1F2D },
	{ 0x1F26, 0x1F2E },
	{ 0x1F27, 0x1F2F },
	{ 0x1F30, 0x1F38 },
	{ 0x1F31, 0x1F39 },
	{ 0x1F32, 0x1F3A },
	{ 0x1F33, 0x1F3B },
	{ 0x1F34, 0x1F3C },
	{ 0x1F35, 0x1F3D },
	{ 0x1F36, 0x1F3E },
	{ 0x1F37, 0x1F3F },
	{ 0x1F40, 0x1F48 },
	{ 0x1F41, 0x1F49 },
	{ 0x1F42, 0x1F4A },
	{ 0x1F43, 0x1F4B },
	{ 0x1F44, 0x1F4C },
	{ 0x1F45, 0x1F4D },
	{ 0x1F51, 0x1F59 },
	{ 0x1F53, 0x1F5B },
	{ 0x1F55, 0x1F5D },
	{ 0x1F57, 0x1F5F },
	{ 0x1F60, 0x1F68 },
	{ 0x1F61, 0x1F69 },
	{ 0x1F62, 0x1F6A },
	{ 0x1F63, 0x1F6B },
	{ 0x1F64, 0x1F6C },
	{ 0x1F65, 0x1F6D },
	{ 0x1F66, 0x1F6E },
	{ 0x1F67, 0x1F6F },
	{ 0x1F80, 0x1F88 },
	{ 0x1F81, 0x1F89 },
	{ 0x1F82, 0x1F8A },
	{ 0x1F83, 0x1F8B },
	{ 0x1F84, 0x1F8C },
	{ 0x1F85, 0x1F8D },
	{ 0x1F86, 0x1F8E },
	{ 0x1F87, 0x1F8F },
	{ 0x1F90, 0x1F98 },
	{ 0x1F91, 0x1F99 },
	{ 0x1F92, 0x1F9A },
	{ 0x1F93, 0x1F9B },
	{ 0x1F94, 0x1F9C },
	{ 0x1F95, 0x1F9D },
	{ 0x1F96, 0x1F9E },
	{ 0x1F97, 0x1F9F },
	{ 0x1FA0, 0x1FA8 },
	{ 0x1FA1, 0x1FA9 },
	{ 0x1FA2, 0x1FAA },
	{ 0x1FA3, 0x1FAB },
	{ 0x1FA4, 0x1FAC },
	{ 0x1FA5, 0x1FAD },
	{ 0x1FA6, 0x1FAE },
	{ 0x1FA7, 0x1FAF },
	{ 0x1FB0, 0x1FB8 },
	{ 0x1FB1, 0x1FB9 },
	{ 0x1FD0, 0x1FD8 },
	{ 0x1FD1, 0x1FD9 },
	{ 0x1FE0, 0x1FE8 },
	{ 0x1FE1, 0x1FE9 },
	{ 0x24D0, 0x24B6 },
	{ 0x24D1, 0x24B7 },
	{ 0x24D2, 0x24B8 },
	{ 0x24D3, 0x24B9 },
	{ 0x24D4, 0x24BA },
	{ 0x24D5, 0x24BB },
	{ 0x24D6, 0x24BC },
	{ 0x24D7, 0x24BD },
	{ 0x24D8, 0x24BE },
	{ 0x24D9, 0x24BF },
	{ 0x24DA, 0x24C0 },
	{ 0x24DB, 0x24C1 },
	{ 0x24DC, 0x24C2 },
	{ 0x24DD, 0x24C3 },
	{ 0x24DE, 0x24C4 },
	{ 0x24DF, 0x24C5 },
	{ 0x24E0, 0x24C6 },
	{ 0x24E1, 0x24C7 },
	{ 0x24E2, 0x24C8 },
	{ 0x24E3, 0x24C9 },
	{ 0x24E4, 0x24CA },
	{ 0x24E5, 0x24CB },
	{ 0x24E6, 0x24CC },
	{ 0x24E7, 0x24CD },
	{ 0x24E8, 0x24CE },
	{ 0x24E9, 0x24CF },
	{ 0xFF41, 0xFF21 },
	{ 0xFF42, 0xFF22 },
	{ 0xFF43, 0xFF23 },
	{ 0xFF44, 0xFF24 },
	{ 0xFF45, 0xFF25 },
	{ 0xFF46, 0xFF26 },
	{ 0xFF47, 0xFF27 },
	{ 0xFF48, 0xFF28 },
	{ 0xFF49, 0xFF29 },
	{ 0xFF4A, 0xFF2A },
	{ 0xFF4B, 0xFF2B },
	{ 0xFF4C, 0xFF2C },
	{ 0xFF4D, 0xFF2D },
	{ 0xFF4E, 0xFF2E },
	{ 0xFF4F, 0xFF2F },
	{ 0xFF50, 0xFF30 },
	{ 0xFF51, 0xFF31 },
	{ 0xFF52, 0xFF32 },
	{ 0xFF53, 0xFF33 },
	{ 0xFF54, 0xFF34 },
	{ 0xFF55, 0xFF35 },
	{ 0xFF56, 0xFF36 },
	{ 0xFF57, 0xFF37 },
	{ 0xFF58, 0xFF38 },
	{ 0xFF59, 0xFF39 },
	{ 0xFF5A, 0xFF3A },
};

static const int reverse_caps_table[CAPS_LEN - 1][2] = {
	{ 0x0041, 0x0061 },
	{ 0x0042, 0x0062 },
	{ 0x0043, 0x0063 },
	{ 0x0044, 0x0064 },
	{ 0x0045, 0x0065 },
	{ 0x0046, 0x0066 },
	{ 0x0047, 0x0067 },
	{ 0x0048, 0x0068 },
	{ 0x0049, 0x0069 },
	// { 0x0049, 0x0131 }, // dotless I
	{ 0x004A, 0x006A },
	{ 0x004B, 0x006B },
	{ 0x004C, 0x006C },
	{ 0x004D, 0x006D },
	{ 0x004E, 0x006E },
	{ 0x004F, 0x006F },
	{ 0x0050, 0x0070 },
	{ 0x0051, 0x0071 },
	{ 0x0052, 0x0072 },
	{ 0x0053, 0x0073 },
	{ 0x0054, 0x0074 },
	{ 0x0055, 0x0075 },
	{ 0x0056, 0x0076 },
	{ 0x0057, 0x0077 },
	{ 0x0058, 0x0078 },
	{ 0x0059, 0x0079 },
	{ 0x005A, 0x007A },
	{ 0x00C0, 0x00E0 },
	{ 0x00C1, 0x00E1 },
	{ 0x00C2, 0x00E2 },
	{ 0x00C3, 0x00E3 },
	{ 0x00C4, 0x00E4 },
	{ 0x00C5, 0x00E5 },
	{ 0x00C6, 0x00E6 },
	{ 0x00C7, 0x00E7 },
	{ 0x00C8, 0x00E8 },
	{ 0x00C9, 0x00E9 },
	{ 0x00CA, 0x00EA },
	{ 0x00CB, 0x00EB },
	{ 0x00CC, 0x00EC },
	{ 0x00CD, 0x00ED },
	{ 0x00CE, 0x00EE },
	{ 0x00CF, 0x00EF },
	{ 0x00D0, 0x00F0 },
	{ 0x00D1, 0x00F1 },
	{ 0x00D2, 0x00F2 },
	{ 0x00D3, 0x00F3 },
	{ 0x00D4, 0x00F4 },
	{ 0x00D5, 0x00F5 },
	{ 0x00D6, 0x00F6 },
	{ 0x00D8, 0x00F8 },
	{ 0x00D9, 0x00F9 },
	{ 0x00DA, 0x00FA },
	{ 0x00DB, 0x00FB },
	{ 0x00DC, 0x00FC },
	{ 0x00DD, 0x00FD },
	{ 0x00DE, 0x00FE },
	{ 0x0100, 0x0101 },
	{ 0x0102, 0x0103 },
	{ 0x0104, 0x0105 },
	{ 0x0106, 0x0107 },
	{ 0x0108, 0x0109 },
	{ 0x010A, 0x010B },
	{ 0x010C, 0x010D },
	{ 0x010E, 0x010F },
	{ 0x0110, 0x0111 },
	{ 0x0112, 0x0113 },
	{ 0x0114, 0x0115 },
	{ 0x0116, 0x0117 },
	{ 0x0118, 0x0119 },
	{ 0x011A, 0x011B },
	{ 0x011C, 0x011D },
	{ 0x011E, 0x011F },
	{ 0x0120, 0x0121 },
	{ 0x0122, 0x0123 },
	{ 0x0124, 0x0125 },
	{ 0x0126, 0x0127 },
	{ 0x0128, 0x0129 },
	{ 0x012A, 0x012B },
	{ 0x012C, 0x012D },
	{ 0x012E, 0x012F },
	{ 0x0132, 0x0133 },
	{ 0x0134, 0x0135 },
	{ 0x0136, 0x0137 },
	{ 0x0139, 0x013A },
	{ 0x013B, 0x013C },
	{ 0x013D, 0x013E },
	{ 0x013F, 0x0140 },
	{ 0x0141, 0x0142 },
	{ 0x0143, 0x0144 },
	{ 0x0145, 0x0146 },
	{ 0x0147, 0x0148 },
	{ 0x014A, 0x014B },
	{ 0x014C, 0x014D },
	{ 0x014E, 0x014F },
	{ 0x0150, 0x0151 },
	{ 0x0152, 0x0153 },
	{ 0x0154, 0x0155 },
	{ 0x0156, 0x0157 },
	{ 0x0158, 0x0159 },
	{ 0x015A, 0x015B },
	{ 0x015C, 0x015D },
	{ 0x015E, 0x015F },
	{ 0x0160, 0x0161 },
	{ 0x0162, 0x0163 },
	{ 0x0164, 0x0165 },
	{ 0x0166, 0x0167 },
	{ 0x0168, 0x0169 },
	{ 0x016A, 0x016B },
	{ 0x016C, 0x016D },
	{ 0x016E, 0x016F },
	{ 0x0170, 0x0171 },
	{ 0x0172, 0x0173 },
	{ 0x0174, 0x0175 },
	{ 0x0176, 0x0177 },
	{ 0x0178, 0x00FF },
	{ 0x0179, 0x017A },
	{ 0x017B, 0x017C },
	{ 0x017D, 0x017E },
	{ 0x0181, 0x0253 },
	{ 0x0182, 0x0183 },
	{ 0x0184, 0x0185 },
	{ 0x0186, 0x0254 },
	{ 0x0187, 0x0188 },
	{ 0x018A, 0x0257 },
	{ 0x018B, 0x018C },
	{ 0x018E, 0x0258 },
	{ 0x018F, 0x0259 },
	{ 0x0190, 0x025B },
	{ 0x0191, 0x0192 },
	{ 0x0193, 0x0260 },
	{ 0x0194, 0x0263 },
	{ 0x0196, 0x0269 },
	{ 0x0197, 0x0268 },
	{ 0x0198, 0x0199 },
	{ 0x019C, 0x026F },
	{ 0x019D, 0x0272 },
	{ 0x019F, 0x0275 },
	{ 0x01A0, 0x01A1 },
	{ 0x01A2, 0x01A3 },
	{ 0x01A4, 0x01A5 },
	{ 0x01A7, 0x01A8 },
	{ 0x01A9, 0x0283 },
	{ 0x01AC, 0x01AD },
	{ 0x01AE, 0x0288 },
	{ 0x01AF, 0x01B0 },
	{ 0x01B1, 0x028A },
	{ 0x01B2, 0x028B },
	{ 0x01B3, 0x01B4 },
	{ 0x01B5, 0x01B6 },
	{ 0x01B7, 0x0292 },
	{ 0x01B8, 0x01B9 },
	{ 0x01BC, 0x01BD },
	{ 0x01C4, 0x01C6 },
	{ 0x01C7, 0x01C9 },
	{ 0x01CA, 0x01CC },
	{ 0x01CD, 0x01CE },
	{ 0x01CF, 0x01D0 },
	{ 0x01D1, 0x01D2 },
	{ 0x01D3, 0x01D4 },
	{ 0x01D5, 0x01D6 },
	{ 0x01D7, 0x01D8 },
	{ 0x01D9, 0x01DA },
	{ 0x01DB, 0x01DC },
	{ 0x01DE, 0x01DF },
	{ 0x01E0, 0x01E1 },
	{ 0x01E2, 0x01E3 },
	{ 0x01E4, 0x01E5 },
	{ 0x01E6, 0x01E7 },
	{ 0x01E8, 0x01E9 },
	{ 0x01EA, 0x01EB },
	{ 0x01EC, 0x01ED },
	{ 0x01EE, 0x01EF },
	{ 0x01F1, 0x01F3 },
	{ 0x01F4, 0x01F5 },
	{ 0x01FA, 0x01FB },
	{ 0x01FC, 0x01FD },
	{ 0x01FE, 0x01FF },
	{ 0x0200, 0x0201 },
	{ 0x0202, 0x0203 },
	{ 0x0204, 0x0205 },
	{ 0x0206, 0x0207 },
	{ 0x0208, 0x0209 },
	{ 0x020A, 0x020B },
	{ 0x020C, 0x020D },
	{ 0x020E, 0x020F },
	{ 0x0210, 0x0211 },
	{ 0x0212, 0x0213 },
	{ 0x0214, 0x0215 },
	{ 0x0216, 0x0217 },
	{ 0x0386, 0x03AC },
	{ 0x0388, 0x03AD },
	{ 0x0389, 0x03AE },
	{ 0x038A, 0x03AF },
	{ 0x038C, 0x03CC },
	{ 0x038E, 0x03CD },
	{ 0x038F, 0x03CE },
	{ 0x0391, 0x03B1 },
	{ 0x0392, 0x03B2 },
	{ 0x0393, 0x03B3 },
	{ 0x0394, 0x03B4 },
	{ 0x0395, 0x03B5 },
	{ 0x0396, 0x03B6 },
	{ 0x0397, 0x03B7 },
	{ 0x0398, 0x03B8 },
	{ 0x0399, 0x03B9 },
	{ 0x039A, 0x03BA },
	{ 0x039B, 0x03BB },
	{ 0x039C, 0x03BC },
	{ 0x039D, 0x03BD },
	{ 0x039E, 0x03BE },
	{ 0x039F, 0x03BF },
	{ 0x03A0, 0x03C0 },
	{ 0x03A1, 0x03C1 },
	{ 0x03A3, 0x03C3 },
	{ 0x03A4, 0x03C4 },
	{ 0x03A5, 0x03C5 },
	{ 0x03A6, 0x03C6 },
	{ 0x03A7, 0x03C7 },
	{ 0x03A8, 0x03C8 },
	{ 0x03A9, 0x03C9 },
	{ 0x03AA, 0x03CA },
	{ 0x03AB, 0x03CB },
	{ 0x03E2, 0x03E3 },
	{ 0x03E4, 0x03E5 },
	{ 0x03E6, 0x03E7 },
	{ 0x03E8, 0x03E9 },
	{ 0x03EA, 0x03EB },
	{ 0x03EC, 0x03ED },
	{ 0x03EE, 0x03EF },
	{ 0x0401, 0x0451 },
	{ 0x0402, 0x0452 },
	{ 0x0403, 0x0453 },
	{ 0x0404, 0x0454 },
	{ 0x0405, 0x0455 },
	{ 0x0406, 0x0456 },
	{ 0x0407, 0x0457 },
	{ 0x0408, 0x0458 },
	{ 0x0409, 0x0459 },
	{ 0x040A, 0x045A },
	{ 0x040B, 0x045B },
	{ 0x040C, 0x045C },
	{ 0x040E, 0x045E },
	{ 0x040F, 0x045F },
	{ 0x0410, 0x0430 },
	{ 0x0411, 0x0431 },
	{ 0x0412, 0x0432 },
	{ 0x0413, 0x0433 },
	{ 0x0414, 0x0434 },
	{ 0x0415, 0x0435 },
	{ 0x0416, 0x0436 },
	{ 0x0417, 0x0437 },
	{ 0x0418, 0x0438 },
	{ 0x0419, 0x0439 },
	{ 0x041A, 0x043A },
	{ 0x041B, 0x043B },
	{ 0x041C, 0x043C },
	{ 0x041D, 0x043D },
	{ 0x041E, 0x043E },
	{ 0x041F, 0x043F },
	{ 0x0420, 0x0440 },
	{ 0x0421, 0x0441 },
	{ 0x0422, 0x0442 },
	{ 0x0423, 0x0443 },
	{ 0x0424, 0x0444 },
	{ 0x0425, 0x0445 },
	{ 0x0426, 0x0446 },
	{ 0x0427, 0x0447 },
	{ 0x0428, 0x0448 },
	{ 0x0429, 0x0449 },
	{ 0x042A, 0x044A },
	{ 0x042B, 0x044B },
	{ 0x042C, 0x044C },
	{ 0x042D, 0x044D },
	{ 0x042E, 0x044E },
	{ 0x042F, 0x044F },
	{ 0x0460, 0x0461 },
	{ 0x0462, 0x0463 },
	{ 0x0464, 0x0465 },
	{ 0x0466, 0x0467 },
	{ 0x0468, 0x0469 },
	{ 0x046A, 0x046B },
	{ 0x046C, 0x046D },
	{ 0x046E, 0x046F },
	{ 0x0470, 0x0471 },
	{ 0x0472, 0x0473 },
	{ 0x0474, 0x0475 },
	{ 0x0476, 0x0477 },
	{ 0x0478, 0x0479 },
	{ 0x047A, 0x047B },
	{ 0x047C, 0x047D },
	{ 0x047E, 0x047F },
	{ 0x0480, 0x0481 },
	{ 0x0490, 0x0491 },
	{ 0x0492, 0x0493 },
	{ 0x0494, 0x0495 },
	{ 0x0496, 0x0497 },
	{ 0x0498, 0x0499 },
	{ 0x049A, 0x049B },
	{ 0x049C, 0x049D },
	{ 0x049E, 0x049F },
	{ 0x04A0, 0x04A1 },
	{ 0x04A2, 0x04A3 },
	{ 0x04A4, 0x04A5 },
	{ 0x04A6, 0x04A7 },
	{ 0x04A8, 0x04A9 },
	{ 0x04AA, 0x04AB },
	{ 0x04AC, 0x04AD },
	{ 0x04AE, 0x04AF },
	{ 0x04B0, 0x04B1 },
	{ 0x04B2, 0x04B3 },
	{ 0x04B4, 0x04B5 },
	{ 0x04B6, 0x04B7 },
	{ 0x04B8, 0x04B9 },
	{ 0x04BA, 0x04BB },
	{ 0x04BC, 0x04BD },
	{ 0x04BE, 0x04BF },
	{ 0x04C1, 0x04C2 },
	{ 0x04C3, 0x04C4 },
	{ 0x04C7, 0x04C8 },
	{ 0x04CB, 0x04CC },
	{ 0x04D0, 0x04D1 },
	{ 0x04D2, 0x04D3 },
	{ 0x04D4, 0x04D5 },
	{ 0x04D6, 0x04D7 },
	{ 0x04D8, 0x04D9 },
	{ 0x04DA, 0x04DB },
	{ 0x04DC, 0x04DD },
	{ 0x04DE, 0x04DF },
	{ 0x04E0, 0x04E1 },
	{ 0x04E2, 0x04E3 },
	{ 0x04E4, 0x04E5 },
	{ 0x04E6, 0x04E7 },
	{ 0x04E8, 0x04E9 },
	{ 0x04EA, 0x04EB },
	{ 0x04EE, 0x04EF },
	{ 0x04F0, 0x04F1 },
	{ 0x04F2, 0x04F3 },
	{ 0x04F4, 0x04F5 },
	{ 0x04F8, 0x04F9 },
	{ 0x0531, 0x0561 },
	{ 0x0532, 0x0562 },
	{ 0x0533, 0x0563 },
	{ 0x0534, 0x0564 },
	{ 0x0535, 0x0565 },
	{ 0x0536, 0x0566 },
	{ 0x0537, 0x0567 },
	{ 0x0538, 0x0568 },
	{ 0x0539, 0x0569 },
	{ 0x053A, 0x056A },
	{ 0x053B, 0x056B },
	{ 0x053C, 0x056C },
	{ 0x053D, 0x056D },
	{ 0x053E, 0x056E },
	{ 0x053F, 0x056F },
	{ 0x0540, 0x0570 },
	{ 0x0541, 0x0571 },
	{ 0x0542, 0x0572 },
	{ 0x0543, 0x0573 },
	{ 0x0544, 0x0574 },
	{ 0x0545, 0x0575 },
	{ 0x0546, 0x0576 },
	{ 0x0547, 0x0577 },
	{ 0x0548, 0x0578 },
	{ 0x0549, 0x0579 },
	{ 0x054A, 0x057A },
	{ 0x054B, 0x057B },
	{ 0x054C, 0x057C },
	{ 0x054D, 0x057D },
	{ 0x054E, 0x057E },
	{ 0x054F, 0x057F },
	{ 0x0550, 0x0580 },
	{ 0x0551, 0x0581 },
	{ 0x0552, 0x0582 },
	{ 0x0553, 0x0583 },
	{ 0x0554, 0x0584 },
	{ 0x0555, 0x0585 },
	{ 0x0556, 0x0586 },
	{ 0x10A0, 0x10D0 },
	{ 0x10A1, 0x10D1 },
	{ 0x10A2, 0x10D2 },
	{ 0x10A3, 0x10D3 },
	{ 0x10A4, 0x10D4 },
	{ 0x10A5, 0x10D5 },
	{ 0x10A6, 0x10D6 },
	{ 0x10A7, 0x10D7 },
	{ 0x10A8, 0x10D8 },
	{ 0x10A9, 0x10D9 },
	{ 0x10AA, 0x10DA },
	{ 0x10AB, 0x10DB },
	{ 0x10AC, 0x10DC },
	{ 0x10AD, 0x10DD },
	{ 0x10AE, 0x10DE },
	{ 0x10AF, 0x10DF },
	{ 0x10B0, 0x10E0 },
	{ 0x10B1, 0x10E1 },
	{ 0x10B2, 0x10E2 },
	{ 0x10B3, 0x10E3 },
	{ 0x10B4, 0x10E4 },
	{ 0x10B5, 0x10E5 },
	{ 0x10B6, 0x10E6 },
	{ 0x10B7, 0x10E7 },
	{ 0x10B8, 0x10E8 },
	{ 0x10B9, 0x10E9 },
	{ 0x10BA, 0x10EA },
	{ 0x10BB, 0x10EB },
	{ 0x10BC, 0x10EC },
	{ 0x10BD, 0x10ED },
	{ 0x10BE, 0x10EE },
	{ 0x10BF, 0x10EF },
	{ 0x10C0, 0x10F0 },
	{ 0x10C1, 0x10F1 },
	{ 0x10C2, 0x10F2 },
	{ 0x10C3, 0x10F3 },
	{ 0x10C4, 0x10F4 },
	{ 0x10C5, 0x10F5 },
	{ 0x1E00, 0x1E01 },
	{ 0x1E02, 0x1E03 },
	{ 0x1E04, 0x1E05 },
	{ 0x1E06, 0x1E07 },
	{ 0x1E08, 0x1E09 },
	{ 0x1E0A, 0x1E0B },
	{ 0x1E0C, 0x1E0D },
	{ 0x1E0E, 0x1E0F },
	{ 0x1E10, 0x1E11 },
	{ 0x1E12, 0x1E13 },
	{ 0x1E14, 0x1E15 },
	{ 0x1E16, 0x1E17 },
	{ 0x1E18, 0x1E19 },
	{ 0x1E1A, 0x1E1B },
	{ 0x1E1C, 0x1E1D },
	{ 0x1E1E, 0x1E1F },
	{ 0x1E20, 0x1E21 },
	{ 0x1E22, 0x1E23 },
	{ 0x1E24, 0x1E25 },
	{ 0x1E26, 0x1E27 },
	{ 0x1E28, 0x1E29 },
	{ 0x1E2A, 0x1E2B },
	{ 0x1E2C, 0x1E2D },
	{ 0x1E2E, 0x1E2F },
	{ 0x1E30, 0x1E31 },
	{ 0x1E32, 0x1E33 },
	{ 0x1E34, 0x1E35 },
	{ 0x1E36, 0x1E37 },
	{ 0x1E38, 0x1E39 },
	{ 0x1E3A, 0x1E3B },
	{ 0x1E3C, 0x1E3D },
	{ 0x1E3E, 0x1E3F },
	{ 0x1E40, 0x1E41 },
	{ 0x1E42, 0x1E43 },
	{ 0x1E44, 0x1E45 },
	{ 0x1E46, 0x1E47 },
	{ 0x1E48, 0x1E49 },
	{ 0x1E4A, 0x1E4B },
	{ 0x1E4C, 0x1E4D },
	{ 0x1E4E, 0x1E4F },
	{ 0x1E50, 0x1E51 },
	{ 0x1E52, 0x1E53 },
	{ 0x1E54, 0x1E55 },
	{ 0x1E56, 0x1E57 },
	{ 0x1E58, 0x1E59 },
	{ 0x1E5A, 0x1E5B },
	{ 0x1E5C, 0x1E5D },
	{ 0x1E5E, 0x1E5F },
	{ 0x1E60, 0x1E61 },
	{ 0x1E62, 0x1E63 },
	{ 0x1E64, 0x1E65 },
	{ 0x1E66, 0x1E67 },
	{ 0x1E68, 0x1E69 },
	{ 0x1E6A, 0x1E6B },
	{ 0x1E6C, 0x1E6D },
	{ 0x1E6E, 0x1E6F },
	{ 0x1E70, 0x1E71 },
	{ 0x1E72, 0x1E73 },
	{ 0x1E74, 0x1E75 },
	{ 0x1E76, 0x1E77 },
	{ 0x1E78, 0x1E79 },
	{ 0x1E7A, 0x1E7B },
	{ 0x1E7C, 0x1E7D },
	{ 0x1E7E, 0x1E7F },
	{ 0x1E80, 0x1E81 },
	{ 0x1E82, 0x1E83 },
	{ 0x1E84, 0x1E85 },
	{ 0x1E86, 0x1E87 },
	{ 0x1E88, 0x1E89 },
	{ 0x1E8A, 0x1E8B },
	{ 0x1E8C, 0x1E8D },
	{ 0x1E8E, 0x1E8F },
	{ 0x1E90, 0x1E91 },
	{ 0x1E92, 0x1E93 },
	{ 0x1E94, 0x1E95 },
	{ 0x1EA0, 0x1EA1 },
	{ 0x1EA2, 0x1EA3 },
	{ 0x1EA4, 0x1EA5 },
	{ 0x1EA6, 0x1EA7 },
	{ 0x1EA8, 0x1EA9 },
	{ 0x1EAA, 0x1EAB },
	{ 0x1EAC, 0x1EAD },
	{ 0x1EAE, 0x1EAF },
	{ 0x1EB0, 0x1EB1 },
	{ 0x1EB2, 0x1EB3 },
	{ 0x1EB4, 0x1EB5 },
	{ 0x1EB6, 0x1EB7 },
	{ 0x1EB8, 0x1EB9 },
	{ 0x1EBA, 0x1EBB },
	{ 0x1EBC, 0x1EBD },
	{ 0x1EBE, 0x1EBF },
	{ 0x1EC0, 0x1EC1 },
	{ 0x1EC2, 0x1EC3 },
	{ 0x1EC4, 0x1EC5 },
	{ 0x1EC6, 0x1EC7 },
	{ 0x1EC8, 0x1EC9 },
	{ 0x1ECA, 0x1ECB },
	{ 0x1ECC, 0x1ECD },
	{ 0x1ECE, 0x1ECF },
	{ 0x1ED0, 0x1ED1 },
	{ 0x1ED2, 0x1ED3 },
	{ 0x1ED4, 0x1ED5 },
	{ 0x1ED6, 0x1ED7 },
	{ 0x1ED8, 0x1ED9 },
	{ 0x1EDA, 0x1EDB },
	{ 0x1EDC, 0x1EDD },
	{ 0x1EDE, 0x1EDF },
	{ 0x1EE0, 0x1EE1 },
	{ 0x1EE2, 0x1EE3 },
	{ 0x1EE4, 0x1EE5 },
	{ 0x1EE6, 0x1EE7 },
	{ 0x1EE8, 0x1EE9 },
	{ 0x1EEA, 0x1EEB },
	{ 0x1EEC, 0x1EED },
	{ 0x1EEE, 0x1EEF },
	{ 0x1EF0, 0x1EF1 },
	{ 0x1EF2, 0x1EF3 },
	{ 0x1EF4, 0x1EF5 },
	{ 0x1EF6, 0x1EF7 },
	{ 0x1EF8, 0x1EF9 },
	{ 0x1F08, 0x1F00 },
	{ 0x1F09, 0x1F01 },
	{ 0x1F0A, 0x1F02 },
	{ 0x1F0B, 0x1F03 },
	{ 0x1F0C, 0x1F04 },
	{ 0x1F0D, 0x1F05 },
	{ 0x1F0E, 0x1F06 },
	{ 0x1F0F, 0x1F07 },
	{ 0x1F18, 0x1F10 },
	{ 0x1F19, 0x1F11 },
	{ 0x1F1A, 0x1F12 },
	{ 0x1F1B, 0x1F13 },
	{ 0x1F1C, 0x1F14 },
	{ 0x1F1D, 0x1F15 },
	{ 0x1F28, 0x1F20 },
	{ 0x1F29, 0x1F21 },
	{ 0x1F2A, 0x1F22 },
	{ 0x1F2B, 0x1F23 },
	{ 0x1F2C, 0x1F24 },
	{ 0x1F2D, 0x1F25 },
	{ 0x1F2E, 0x1F26 },
	{ 0x1F2F, 0x1F27 },
	{ 0x1F38, 0x1F30 },
	{ 0x1F39, 0x1F31 },
	{ 0x1F3A, 0x1F32 },
	{ 0x1F3B, 0x1F33 },
	{ 0x1F3C, 0x1F34 },
	{ 0x1F3D, 0x1F35 },
	{ 0x1F3E, 0x1F36 },
	{ 0x1F3F, 0x1F37 },
	{ 0x1F48, 0x1F40 },
	{ 0x1F49, 0x1F41 },
	{ 0x1F4A, 0x1F42 },
	{ 0x1F4B, 0x1F43 },
	{ 0x1F4C, 0x1F44 },
	{ 0x1F4D, 0x1F45 },
	{ 0x1F59, 0x1F51 },
	{ 0x1F5B, 0x1F53 },
	{ 0x1F5D, 0x1F55 },
	{ 0x1F5F, 0x1F57 },
	{ 0x1F68, 0x1F60 },
	{ 0x1F69, 0x1F61 },
	{ 0x1F6A, 0x1F62 },
	{ 0x1F6B, 0x1F63 },
	{ 0x1F6C, 0x1F64 },
	{ 0x1F6D, 0x1F65 },
	{ 0x1F6E, 0x1F66 },
	{ 0x1F6F, 0x1F67 },
	{ 0x1F88, 0x1F80 },
	{ 0x1F89, 0x1F81 },
	{ 0x1F8A, 0x1F82 },
	{ 0x1F8B, 0x1F83 },
	{ 0x1F8C, 0x1F84 },
	{ 0x1F8D, 0x1F85 },
	{ 0x1F8E, 0x1F86 },
	{ 0x1F8F, 0x1F87 },
	{ 0x1F98, 0x1F90 },
	{ 0x1F99, 0x1F91 },
	{ 0x1F9A, 0x1F92 },
	{ 0x1F9B, 0x1F93 },
	{ 0x1F9C, 0x1F94 },
	{ 0x1F9D, 0x1F95 },
	{ 0x1F9E, 0x1F96 },
	{ 0x1F9F, 0x1F97 },
	{ 0x1FA8, 0x1FA0 },
	{ 0x1FA9, 0x1FA1 },
	{ 0x1FAA, 0x1FA2 },
	{ 0x1FAB, 0x1FA3 },
	{ 0x1FAC, 0x1FA4 },
	{ 0x1FAD, 0x1FA5 },
	{ 0x1FAE, 0x1FA6 },
	{ 0x1FAF, 0x1FA7 },
	{ 0x1FB8, 0x1FB0 },
	{ 0x1FB9, 0x1FB1 },
	{ 0x1FD8, 0x1FD0 },
	{ 0x1FD9, 0x1FD1 },
	{ 0x1FE8, 0x1FE0 },
	{ 0x1FE9, 0x1FE1 },
	{ 0x24B6, 0x24D0 },
	{ 0x24B7, 0x24D1 },
	{ 0x24B8, 0x24D2 },
	{ 0x24B9, 0x24D3 },
	{ 0x24BA, 0x24D4 },
	{ 0x24BB, 0x24D5 },
	{ 0x24BC, 0x24D6 },
	{ 0x24BD, 0x24D7 },
	{ 0x24BE, 0x24D8 },
	{ 0x24BF, 0x24D9 },
	{ 0x24C0, 0x24DA },
	{ 0x24C1, 0x24DB },
	{ 0x24C2, 0x24DC },
	{ 0x24C3, 0x24DD },
	{ 0x24C4, 0x24DE },
	{ 0x24C5, 0x24DF },
	{ 0x24C6, 0x24E0 },
	{ 0x24C7, 0x24E1 },
	{ 0x24C8, 0x24E2 },
	{ 0x24C9, 0x24E3 },
	{ 0x24CA, 0x24E4 },
	{ 0x24CB, 0x24E5 },
	{ 0x24CC, 0x24E6 },
	{ 0x24CD, 0x24E7 },
	{ 0x24CE, 0x24E8 },
	{ 0x24CF, 0x24E9 },
	{ 0xFF21, 0xFF41 },
	{ 0xFF22, 0xFF42 },
	{ 0xFF23, 0xFF43 },
	{ 0xFF24, 0xFF44 },
	{ 0xFF25, 0xFF45 },
	{ 0xFF26, 0xFF46 },
	{ 0xFF27, 0xFF47 },
	{ 0xFF28, 0xFF48 },
	{ 0xFF29, 0xFF49 },
	{ 0xFF2A, 0xFF4A },
	{ 0xFF2B, 0xFF4B },
	{ 0xFF2C, 0xFF4C },
	{ 0xFF2D, 0xFF4D },
	{ 0xFF2E, 0xFF4E },
	{ 0xFF2F, 0xFF4F },
	{ 0xFF30, 0xFF50 },
	{ 0xFF31, 0xFF51 },
	{ 0xFF32, 0xFF52 },
	{ 0xFF33, 0xFF53 },
	{ 0xFF34, 0xFF54 },
	{ 0xFF35, 0xFF55 },
	{ 0xFF36, 0xFF56 },
	{ 0xFF37, 0xFF57 },
	{ 0xFF38, 0xFF58 },
	{ 0xFF39, 0xFF59 },
	{ 0xFF3A, 0xFF5A },
};

static int _find_upper(int ch) {
	int low = 0;
	int high = CAPS_LEN - 1;
	int middle;

	while (low <= high) {
		middle = (low + high) / 2;

		if (ch < caps_table[middle][0]) {
			high = middle - 1; //search low end of array
		} else if (caps_table[middle][0] < ch) {
			low = middle + 1; //search high end of array
		} else {
			return caps_table[middle][1];
		}
	}

	return ch;
}

static int _find_lower(int ch) {
	int low = 0;
	int high = CAPS_LEN - 2;
	int middle;

	while (low <= high) {
		middle = (low + high) / 2;

		if (ch < reverse_caps_table[middle][0]) {
			high = middle - 1; //search low end of array
		} else if (reverse_caps_table[middle][0] < ch) {
			low = middle + 1; //search high end of array
		} else {
			return reverse_caps_table[middle][1];
		}
	}

	return ch;
}

#line 0

#line 1 "sfw/core/mutex.cpp"
/*************************************************************************/
/*  mutex.cpp                                                            */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

static Mutex _global_mutex;

void _global_lock() {
	_global_mutex.lock();
}

void _global_unlock() {
	_global_mutex.unlock();
}

#ifndef NO_THREADS

template class MutexImpl<std::recursive_mutex>;
template class MutexImpl<std::mutex>;

#endif
#line 0

#line 1 "sfw/core/safe_refcount.cpp"
/*************************************************************************/
/*  safe_refcount.cpp                                                    */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

#if defined(DEBUG_ENABLED) && !defined(NO_THREADS)

// On C++14 we don't have std::atomic::is_always_lockfree, so this is the best we can do
void check_lockless_atomics() {
	// Doing the check for the types we actually care about
	if (!std::atomic<uint32_t>{}.is_lock_free() || !std::atomic<uint64_t>{}.is_lock_free() || !std::atomic_bool{}.is_lock_free()) {
		LOG_WARN("Your compiler doesn't seem to support lockless atomics. Performance will be degraded. Please consider upgrading to a different or newer compiler.");
	}
}

#endif
#line 0

//#include "thread.h"
//#include "core/error_macros.h"
//#include "core/safe_refcount.h"

#line 1 "sfw/core/thread.cpp"
/*************************************************************************/
/*  thread.cpp                                                           */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

#if !defined(NO_THREADS)

Error (*Thread::set_name_func)(const String &) = nullptr;
void (*Thread::set_priority_func)(Thread::Priority) = nullptr;
void (*Thread::init_func)() = nullptr;
void (*Thread::term_func)() = nullptr;

uint64_t Thread::_thread_id_hash(const std::thread::id &p_t) {
	static std::hash<std::thread::id> hasher;
	return hasher(p_t);
}

Thread::ID Thread::main_thread_id = _thread_id_hash(std::this_thread::get_id());
static thread_local Thread::ID caller_id = 0;
static thread_local bool caller_id_cached = false;

void Thread::_set_platform_funcs(
		Error (*p_set_name_func)(const String &),
		void (*p_set_priority_func)(Thread::Priority),
		void (*p_init_func)(),
		void (*p_term_func)()) {
	Thread::set_name_func = p_set_name_func;
	Thread::set_priority_func = p_set_priority_func;
	Thread::init_func = p_init_func;
	Thread::term_func = p_term_func;
}

void Thread::callback(Thread *p_self, const Settings &p_settings, Callback p_callback, void *p_userdata) {
	caller_id = _thread_id_hash(p_self->thread.get_id());
	caller_id_cached = true;

	if (set_priority_func) {
		set_priority_func(p_settings.priority);
	}
	if (init_func) {
		init_func();
	}

	p_callback(p_userdata);

	if (term_func) {
		term_func();
	}
}

void Thread::start(Thread::Callback p_callback, void *p_user, const Settings &p_settings) {
	if (id != _thread_id_hash(std::thread::id())) {
#ifdef DEBUG_ENABLED
		LOG_WARN("A Thread object has been re-started without wait_to_finish() having been called on it. Please do so to ensure correct cleanup of the thread.");
#endif
		thread.detach();
		std::thread empty_thread;
		thread.swap(empty_thread);
	}
	std::thread new_thread(&Thread::callback, this, p_settings, p_callback, p_user);
	thread.swap(new_thread);
	id = _thread_id_hash(thread.get_id());
}

bool Thread::is_started() const {
	return id != _thread_id_hash(std::thread::id());
}

void Thread::wait_to_finish() {
	if (id != _thread_id_hash(std::thread::id())) {
		ERR_FAIL_COND_MSG(id == get_caller_id(), "A Thread can't wait for itself to finish.");
		thread.join();
		std::thread empty_thread;
		thread.swap(empty_thread);
		id = _thread_id_hash(std::thread::id());
	}
}

Error Thread::set_name(const String &p_name) {
	if (set_name_func) {
		return set_name_func(p_name);
	}

	return ERR_UNAVAILABLE;
}

Thread::~Thread() {
	if (id != _thread_id_hash(std::thread::id())) {
#ifdef DEBUG_ENABLED
		LOG_WARN("A Thread object has been destroyed without wait_to_finish() having been called on it. Please do so to ensure correct cleanup of the thread.");
#endif
		thread.detach();
	}
}

Thread::ID Thread::get_caller_id() {
	if (likely(caller_id_cached)) {
		return caller_id;
	} else {
		caller_id = _thread_id_hash(std::this_thread::get_id());
		caller_id_cached = true;
		return caller_id;
	}
}
#endif
#line 0

#line 1 "sfw/core/sfw_time.cpp"
// ----------------------------------------------------------------------------
// time

#if defined(_WIN64) || defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winnt.h>
#else
#include <sys/time.h>
#include <unistd.h>
#endif

#if 0
uint64_t SFWTime::time_gpu() {
    GLint64 t = 123456789;
    glGetInteger64v(GL_TIMESTAMP, &t);
    return (uint64_t)t;
}
#endif
uint64_t SFWTime::date() {
	//time_t epoch = time(0);
	//struct tm *ti = localtime(&epoch);
	//return atoi64(va("%04d%02d%02d%02d%02d%02d", ti->tm_year + 1900, ti->tm_mon + 1, ti->tm_mday, ti->tm_hour, ti->tm_min, ti->tm_sec));
	return 0;
}
char *SFWTime::date_string() {
	//time_t epoch = time(0);
	//struct tm *ti = localtime(&epoch);
	//return va("%04d-%02d-%02d %02d:%02d:%02d", ti->tm_year + 1900, ti->tm_mon + 1, ti->tm_mday, ti->tm_hour, ti->tm_min, ti->tm_sec);
	return 0;
}
uint64_t SFWTime::date_epoch() {
	time_t epoch = time(0);
	return epoch;
}
#if 0
double SFWTime::time_ss() {
    return glfwGetTime();
}
double SFWTime::time_ms() {
    return glfwGetTime() * 1000.0;
}
uint64_t SFWTime::time_us() {
    return (uint64_t)(glfwGetTime() * 1000000.0); // @fixme: use a high resolution timer instead, or time_gpu below
}
uint64_t SFWTime::sleep_us(uint64_t us) { // @fixme: use a high resolution sleeper instead
    return sleep_ms( us / 1000.0 );
}
double SFWTime::sleep_ms(double ms) {
    double now = time_ms();
    if( ms <= 0 ) {
#if defined(_WIN64) || defined(_WIN32)
        Sleep(0); // yield
#else
        usleep(0);
#endif
    } else {
#if defined(_WIN64) || defined(_WIN32)
        Sleep(ms);
#else
        usleep(ms * 1000);
#endif
    }
    return time_ms() - now;
}
double sleep_ss(double ss) {
    return sleep_ms( ss * 1000 ) / 1000.0;
}
#endif

// high-perf functions

#define TIMER_E3 1000ULL
#define TIMER_E6 1000000ULL
#define TIMER_E9 1000000000ULL

#ifdef CLOCK_MONOTONIC_RAW
#define TIME_MONOTONIC CLOCK_MONOTONIC_RAW
#elif defined CLOCK_MONOTONIC
#define TIME_MONOTONIC CLOCK_MONOTONIC
#else
// #define TIME_MONOTONIC CLOCK_REALTIME // untested
#endif

static uint64_t nanotimer(uint64_t *out_freq) {
	if (out_freq) {
#if defined(_WIN64) || defined(_WIN32)
		LARGE_INTEGER li;
		QueryPerformanceFrequency(&li);
		*out_freq = li.QuadPart;
//#elif is(ANDROID)
//      *out_freq = CLOCKS_PER_SEC;
#elif defined TIME_MONOTONIC
		*out_freq = TIMER_E9;
#else
		*out_freq = TIMER_E6;
#endif
	}
#if defined(_WIN64) || defined(_WIN32)
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	return (uint64_t)li.QuadPart;
//#elif is(ANDROID)
//    return (uint64_t)clock();
#elif defined TIME_MONOTONIC
	struct timespec ts;
	clock_gettime(TIME_MONOTONIC, &ts);
	return (TIMER_E9 * (uint64_t)ts.tv_sec) + ts.tv_nsec;
#else
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (TIMER_E6 * (uint64_t)tv.tv_sec) + tv.tv_usec;
#endif
}

uint64_t SFWTime::time_ns() {
	static thread_local uint64_t epoch = 0;
	static thread_local uint64_t freq = 0;
	if (!freq) {
		epoch = nanotimer(&freq);
	}

	uint64_t a = nanotimer(NULL) - epoch;
	uint64_t b = TIMER_E9;
	uint64_t c = freq;

	// Computes (a*b)/c without overflow, as long as both (a*b) and the overall result fit into 64-bits.
	// [ref] https://github.com/rust-lang/rust/blob/3809bbf47c8557bd149b3e52ceb47434ca8378d5/src/libstd/sys_common/mod.rs#L124
	uint64_t q = a / c;
	uint64_t r = a % c;
	return q * b + r * b / c;
}
uint64_t SFWTime::time_us() {
	return time_ns() / TIMER_E3;
}
uint64_t SFWTime::time_ms() {
	return time_ns() / TIMER_E6;
}
double SFWTime::time_ss() {
	return time_ns() / 1e9; // TIMER_E9;
}
double SFWTime::time_mm() {
	return time_ss() / 60;
}
double SFWTime::time_hh() {
	return time_mm() / 60;
}

void SFWTime::sleep_ns(double ns) {
#if defined(_WIN64) || defined(_WIN32)
	if (ns >= 100) {
		LARGE_INTEGER li; // Windows sleep in 100ns units
		HANDLE timer = CreateWaitableTimer(NULL, TRUE, NULL);
		li.QuadPart = (LONGLONG)(__int64)(-ns / 100); // Negative for relative time
		SetWaitableTimer(timer, &li, 0, NULL, NULL, FALSE);
		WaitForSingleObject(timer, INFINITE);
		CloseHandle(timer);
#else
	if (ns > 0) {
		struct timespec wait = { 0 };
		wait.tv_sec = ns / 1e9;
		wait.tv_nsec = ns - wait.tv_sec * 1e9;
		nanosleep(&wait, NULL);
#endif
	} else {
#if defined(_WIN64) || defined(_WIN32)
		Sleep(0); // yield, Sleep(0), SwitchToThread
#else
		usleep(0);
#endif
	}
}
void SFWTime::sleep_us(double us) {
	sleep_ns(us * 1e3);
}
void SFWTime::sleep_ms(double ms) {
	sleep_ns(ms * 1e6);
}
void SFWTime::sleep_ss(double ss) {
	sleep_ns(ss * 1e9);
}
#line 0

#line 1 "sfw/core/memory.cpp"
/*************************************************************************/
/*  memory.cpp                                                           */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

void *operator new(size_t p_size, const char *p_description) {
	return Memory::alloc_static(p_size, false);
}

void *operator new(size_t p_size, void *(*p_allocfunc)(size_t p_size)) {
	return p_allocfunc(p_size);
}

#ifdef _MSC_VER
void operator delete(void *p_mem, const char *p_description) {
	CRASH_MSG("Call to placement delete should not happen.");
}

void operator delete(void *p_mem, void *(*p_allocfunc)(size_t p_size)) {
	CRASH_MSG("Call to placement delete should not happen.");
}

void operator delete(void *p_mem, void *p_pointer, size_t check, const char *p_description) {
	CRASH_MSG("Call to placement delete should not happen.");
}
#endif

#ifdef DEBUG_ENABLED
SafeNumeric<uint64_t> Memory::mem_usage;
SafeNumeric<uint64_t> Memory::max_usage;
#endif

SafeNumeric<uint64_t> Memory::alloc_count;

void *Memory::alloc_static(size_t p_bytes, bool p_pad_align) {
#ifdef DEBUG_ENABLED
	bool prepad = true;
#else
	bool prepad = p_pad_align;
#endif

	void *mem = malloc(p_bytes + (prepad ? PAD_ALIGN : 0));

	ERR_FAIL_COND_V(!mem, nullptr);

	alloc_count.increment();

	if (prepad) {
		uint64_t *s = (uint64_t *)mem;
		*s = p_bytes;

		uint8_t *s8 = (uint8_t *)mem;

#ifdef DEBUG_ENABLED
		uint64_t new_mem_usage = mem_usage.add(p_bytes);
		max_usage.exchange_if_greater(new_mem_usage);
#endif
		return s8 + PAD_ALIGN;
	} else {
		return mem;
	}
}

void *Memory::realloc_static(void *p_memory, size_t p_bytes, bool p_pad_align) {
	if (p_memory == nullptr) {
		return alloc_static(p_bytes, p_pad_align);
	}

	uint8_t *mem = (uint8_t *)p_memory;

#ifdef DEBUG_ENABLED
	bool prepad = true;
#else
	bool prepad = p_pad_align;
#endif

	if (prepad) {
		mem -= PAD_ALIGN;
		uint64_t *s = (uint64_t *)mem;

#ifdef DEBUG_ENABLED
		if (p_bytes > *s) {
			uint64_t new_mem_usage = mem_usage.add(p_bytes - *s);
			max_usage.exchange_if_greater(new_mem_usage);
		} else {
			mem_usage.sub(*s - p_bytes);
		}
#endif

		if (p_bytes == 0) {
			free(mem);
			return nullptr;
		} else {
			*s = p_bytes;

			mem = (uint8_t *)realloc(mem, p_bytes + PAD_ALIGN);
			ERR_FAIL_COND_V(!mem, nullptr);

			s = (uint64_t *)mem;

			*s = p_bytes;

			return mem + PAD_ALIGN;
		}
	} else {
		mem = (uint8_t *)realloc(mem, p_bytes);

		ERR_FAIL_COND_V(mem == nullptr && p_bytes > 0, nullptr);

		return mem;
	}
}

void Memory::free_static(void *p_ptr, bool p_pad_align) {
	ERR_FAIL_COND(p_ptr == nullptr);

	uint8_t *mem = (uint8_t *)p_ptr;

#ifdef DEBUG_ENABLED
	bool prepad = true;
#else
	bool prepad = p_pad_align;
#endif

	alloc_count.decrement();

	if (prepad) {
		mem -= PAD_ALIGN;

#ifdef DEBUG_ENABLED
		uint64_t *s = (uint64_t *)mem;
		mem_usage.sub(*s);
#endif

		free(mem);
	} else {
		free(mem);
	}
}

uint64_t Memory::get_mem_available() {
	return -1; // 0xFFFF...
}

uint64_t Memory::get_mem_usage() {
#ifdef DEBUG_ENABLED
	return mem_usage.get();
#else
	return 0;
#endif
}

uint64_t Memory::get_mem_max_usage() {
#ifdef DEBUG_ENABLED
	return max_usage.get();
#else
	return 0;
#endif
}

_GlobalNil::_GlobalNil() {
	color = 1;
	left = this;
	right = this;
	parent = this;
}

_GlobalNil _GlobalNilClass::_nil;
#line 0

#line 1 "sfw/core/pool_vector.cpp"
/*************************************************************************/
/*  pool_vector.cpp                                                      */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

Mutex pool_vector_lock;

PoolAllocator *MemoryPool::memory_pool = nullptr;
uint8_t *MemoryPool::pool_memory = nullptr;
size_t *MemoryPool::pool_size = nullptr;

MemoryPool::Alloc *MemoryPool::allocs = nullptr;
MemoryPool::Alloc *MemoryPool::free_list = nullptr;
uint32_t MemoryPool::alloc_count = 0;
uint32_t MemoryPool::allocs_used = 0;
Mutex MemoryPool::alloc_mutex;

size_t MemoryPool::total_memory = 0;
size_t MemoryPool::max_memory = 0;

void MemoryPool::setup(uint32_t p_max_allocs) {
	allocs = memnew_arr(Alloc, p_max_allocs);
	alloc_count = p_max_allocs;
	allocs_used = 0;

	for (uint32_t i = 0; i < alloc_count - 1; i++) {
		allocs[i].free_list = &allocs[i + 1];
	}

	free_list = &allocs[0];
}

void MemoryPool::cleanup() {
	memdelete_arr(allocs);

	ERR_FAIL_COND_MSG(allocs_used > 0, "There are still MemoryPool allocs in use at exit!");
}
#line 0

#line 1 "sfw/core/pool_allocator.cpp"
/*************************************************************************/
/*  pool_allocator.cpp                                                   */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

#define COMPACT_CHUNK(m_entry, m_to_pos)                      \
	do {                                                      \
		void *_dst = &((unsigned char *)pool)[m_to_pos];      \
		void *_src = &((unsigned char *)pool)[(m_entry).pos]; \
		memmove(_dst, _src, aligned((m_entry).len));          \
		(m_entry).pos = m_to_pos;                             \
	} while (0);

void PoolAllocator::mt_lock() const {
}

void PoolAllocator::mt_unlock() const {
}

bool PoolAllocator::get_free_entry(EntryArrayPos *p_pos) {
	if (entry_count == entry_max) {
		return false;
	}

	for (int i = 0; i < entry_max; i++) {
		if (entry_array[i].len == 0) {
			*p_pos = i;
			return true;
		}
	}

	ERR_PRINT("Out of memory Chunks!");

	return false; //
}

/**
 * Find a hole
 * @param p_pos The hole is behind the block pointed by this variable upon return. if pos==entry_count, then allocate at end
 * @param p_for_size hole size
 * @return false if hole found, true if no hole found
 */
bool PoolAllocator::find_hole(EntryArrayPos *p_pos, int p_for_size) {
	/* position where previous entry ends. Defaults to zero (begin of pool) */

	int prev_entry_end_pos = 0;

	for (int i = 0; i < entry_count; i++) {
		Entry &entry = entry_array[entry_indices[i]];

		/* determine hole size to previous entry */

		int hole_size = entry.pos - prev_entry_end_pos;

		/* determine if what we want fits in that hole */
		if (hole_size >= p_for_size) {
			*p_pos = i;
			return true;
		}

		/* prepare for next one */
		prev_entry_end_pos = entry_end(entry);
	}

	/* No holes between entries, check at the end..*/

	if ((pool_size - prev_entry_end_pos) >= p_for_size) {
		*p_pos = entry_count;
		return true;
	}

	return false;
}

void PoolAllocator::compact(int p_up_to) {
	uint32_t prev_entry_end_pos = 0;

	if (p_up_to < 0) {
		p_up_to = entry_count;
	}
	for (int i = 0; i < p_up_to; i++) {
		Entry &entry = entry_array[entry_indices[i]];

		/* determine hole size to previous entry */

		int hole_size = entry.pos - prev_entry_end_pos;

		/* if we can compact, do it */
		if (hole_size > 0 && !entry.lock) {
			COMPACT_CHUNK(entry, prev_entry_end_pos);
		}

		/* prepare for next one */
		prev_entry_end_pos = entry_end(entry);
	}
}

void PoolAllocator::compact_up(int p_from) {
	uint32_t next_entry_end_pos = pool_size; // - static_area_size;

	for (int i = entry_count - 1; i >= p_from; i--) {
		Entry &entry = entry_array[entry_indices[i]];

		/* determine hole size to nextious entry */

		int hole_size = next_entry_end_pos - (entry.pos + aligned(entry.len));

		/* if we can compact, do it */
		if (hole_size > 0 && !entry.lock) {
			COMPACT_CHUNK(entry, (next_entry_end_pos - aligned(entry.len)));
		}

		/* prepare for next one */
		next_entry_end_pos = entry.pos;
	}
}

bool PoolAllocator::find_entry_index(EntryIndicesPos *p_map_pos, Entry *p_entry) {
	EntryArrayPos entry_pos = entry_max;

	for (int i = 0; i < entry_count; i++) {
		if (&entry_array[entry_indices[i]] == p_entry) {
			entry_pos = i;
			break;
		}
	}

	if (entry_pos == entry_max) {
		return false;
	}

	*p_map_pos = entry_pos;
	return true;
}

PoolAllocator::ID PoolAllocator::alloc(int p_size) {
	ERR_FAIL_COND_V(p_size < 1, POOL_ALLOCATOR_INVALID_ID);
	ERR_FAIL_COND_V(p_size > free_mem, POOL_ALLOCATOR_INVALID_ID);

	mt_lock();

	if (entry_count == entry_max) {
		mt_unlock();
		ERR_PRINT("entry_count==entry_max");
		return POOL_ALLOCATOR_INVALID_ID;
	}

	int size_to_alloc = aligned(p_size);

	EntryIndicesPos new_entry_indices_pos;

	if (!find_hole(&new_entry_indices_pos, size_to_alloc)) {
		/* No hole could be found, try compacting mem */
		compact();
		/* Then search again */

		if (!find_hole(&new_entry_indices_pos, size_to_alloc)) {
			mt_unlock();
			ERR_FAIL_V_MSG(POOL_ALLOCATOR_INVALID_ID, "Memory can't be compacted further.");
		}
	}

	EntryArrayPos new_entry_array_pos;

	bool found_free_entry = get_free_entry(&new_entry_array_pos);

	if (!found_free_entry) {
		mt_unlock();
		ERR_FAIL_V_MSG(POOL_ALLOCATOR_INVALID_ID, "No free entry found in PoolAllocator.");
	}

	/* move all entry indices up, make room for this one */
	for (int i = entry_count; i > new_entry_indices_pos; i--) {
		entry_indices[i] = entry_indices[i - 1];
	}

	entry_indices[new_entry_indices_pos] = new_entry_array_pos;

	entry_count++;

	Entry &entry = entry_array[entry_indices[new_entry_indices_pos]];

	entry.len = p_size;
	entry.pos = (new_entry_indices_pos == 0) ? 0 : entry_end(entry_array[entry_indices[new_entry_indices_pos - 1]]); //alloc either at beginning or end of previous
	entry.lock = 0;
	entry.check = (check_count++) & CHECK_MASK;
	free_mem -= size_to_alloc;
	if (free_mem < free_mem_peak) {
		free_mem_peak = free_mem;
	}

	ID retval = (entry_indices[new_entry_indices_pos] << CHECK_BITS) | entry.check;
	mt_unlock();

	//ERR_FAIL_COND_V( (uintptr_t)get(retval)%align != 0, retval );

	return retval;
}

PoolAllocator::Entry *PoolAllocator::get_entry(ID p_mem) {
	unsigned int check = p_mem & CHECK_MASK;
	int entry = p_mem >> CHECK_BITS;
	ERR_FAIL_INDEX_V(entry, entry_max, nullptr);
	ERR_FAIL_COND_V(entry_array[entry].check != check, nullptr);
	ERR_FAIL_COND_V(entry_array[entry].len == 0, nullptr);

	return &entry_array[entry];
}

const PoolAllocator::Entry *PoolAllocator::get_entry(ID p_mem) const {
	unsigned int check = p_mem & CHECK_MASK;
	int entry = p_mem >> CHECK_BITS;
	ERR_FAIL_INDEX_V(entry, entry_max, nullptr);
	ERR_FAIL_COND_V(entry_array[entry].check != check, nullptr);
	ERR_FAIL_COND_V(entry_array[entry].len == 0, nullptr);

	return &entry_array[entry];
}

void PoolAllocator::free(ID p_mem) {
	mt_lock();
	Entry *e = get_entry(p_mem);
	if (!e) {
		mt_unlock();
		ERR_PRINT("!e");
		return;
	}
	if (e->lock) {
		mt_unlock();
		ERR_PRINT("e->lock");
		return;
	}

	EntryIndicesPos entry_indices_pos;

	bool index_found = find_entry_index(&entry_indices_pos, e);
	if (!index_found) {
		mt_unlock();
		ERR_FAIL_COND(!index_found);
	}

	for (int i = entry_indices_pos; i < (entry_count - 1); i++) {
		entry_indices[i] = entry_indices[i + 1];
	}

	entry_count--;
	free_mem += aligned(e->len);
	e->clear();
	mt_unlock();
}

int PoolAllocator::get_size(ID p_mem) const {
	int size;
	mt_lock();

	const Entry *e = get_entry(p_mem);
	if (!e) {
		mt_unlock();
		ERR_PRINT("!e");
		return 0;
	}

	size = e->len;

	mt_unlock();

	return size;
}

Error PoolAllocator::resize(ID p_mem, int p_new_size) {
	mt_lock();
	Entry *e = get_entry(p_mem);

	if (!e) {
		mt_unlock();
		ERR_FAIL_COND_V(!e, ERR_INVALID_PARAMETER);
	}

	if (needs_locking && e->lock) {
		mt_unlock();
		ERR_FAIL_COND_V(e->lock, ERR_ALREADY_IN_USE);
	}

	uint32_t alloc_size = aligned(p_new_size);

	if ((uint32_t)aligned(e->len) == alloc_size) {
		e->len = p_new_size;
		mt_unlock();
		return OK;
	} else if (e->len > (uint32_t)p_new_size) {
		free_mem += aligned(e->len);
		free_mem -= alloc_size;
		e->len = p_new_size;
		mt_unlock();
		return OK;
	}

	//p_new_size = align(p_new_size)
	int _free = free_mem; // - static_area_size;

	if (uint32_t(_free + aligned(e->len)) < alloc_size) {
		mt_unlock();
		ERR_FAIL_V(ERR_OUT_OF_MEMORY);
	};

	EntryIndicesPos entry_indices_pos;

	bool index_found = find_entry_index(&entry_indices_pos, e);

	if (!index_found) {
		mt_unlock();
		ERR_FAIL_COND_V(!index_found, ERR_BUG);
	}

	//no need to move stuff around, it fits before the next block
	uint32_t next_pos;
	if (entry_indices_pos + 1 == entry_count) {
		next_pos = pool_size; // - static_area_size;
	} else {
		next_pos = entry_array[entry_indices[entry_indices_pos + 1]].pos;
	};

	if ((next_pos - e->pos) > alloc_size) {
		free_mem += aligned(e->len);
		e->len = p_new_size;
		free_mem -= alloc_size;
		mt_unlock();
		return OK;
	}
	//it doesn't fit, compact around BEFORE current index (make room behind)

	compact(entry_indices_pos + 1);

	if ((next_pos - e->pos) > alloc_size) {
		//now fits! hooray!
		free_mem += aligned(e->len);
		e->len = p_new_size;
		free_mem -= alloc_size;
		mt_unlock();
		if (free_mem < free_mem_peak) {
			free_mem_peak = free_mem;
		}
		return OK;
	}

	//STILL doesn't fit, compact around AFTER current index (make room after)

	compact_up(entry_indices_pos + 1);

	if ((entry_array[entry_indices[entry_indices_pos + 1]].pos - e->pos) > alloc_size) {
		//now fits! hooray!
		free_mem += aligned(e->len);
		e->len = p_new_size;
		free_mem -= alloc_size;
		mt_unlock();
		if (free_mem < free_mem_peak) {
			free_mem_peak = free_mem;
		}
		return OK;
	}

	mt_unlock();
	ERR_FAIL_V(ERR_OUT_OF_MEMORY);
}

Error PoolAllocator::lock(ID p_mem) {
	if (!needs_locking) {
		return OK;
	}
	mt_lock();
	Entry *e = get_entry(p_mem);
	if (!e) {
		mt_unlock();
		ERR_PRINT("!e");
		return ERR_INVALID_PARAMETER;
	}
	e->lock++;
	mt_unlock();
	return OK;
}

bool PoolAllocator::is_locked(ID p_mem) const {
	if (!needs_locking) {
		return false;
	}

	mt_lock();
	const Entry *e = ((PoolAllocator *)(this))->get_entry(p_mem);
	if (!e) {
		mt_unlock();
		ERR_PRINT("!e");
		return false;
	}
	bool locked = e->lock;
	mt_unlock();
	return locked;
}

const void *PoolAllocator::get(ID p_mem) const {
	if (!needs_locking) {
		const Entry *e = get_entry(p_mem);
		ERR_FAIL_COND_V(!e, nullptr);
		return &pool[e->pos];
	}

	mt_lock();
	const Entry *e = get_entry(p_mem);

	if (!e) {
		mt_unlock();
		ERR_FAIL_COND_V(!e, nullptr);
	}
	if (e->lock == 0) {
		mt_unlock();
		ERR_PRINT("e->lock == 0");
		return nullptr;
	}

	if ((int)e->pos >= pool_size) {
		mt_unlock();
		ERR_PRINT("e->pos<0 || e->pos>=pool_size");
		return nullptr;
	}
	const void *ptr = &pool[e->pos];

	mt_unlock();

	return ptr;
}

void *PoolAllocator::get(ID p_mem) {
	if (!needs_locking) {
		Entry *e = get_entry(p_mem);
		ERR_FAIL_COND_V(!e, nullptr);
		return &pool[e->pos];
	}

	mt_lock();
	Entry *e = get_entry(p_mem);

	if (!e) {
		mt_unlock();
		ERR_FAIL_COND_V(!e, nullptr);
	}
	if (e->lock == 0) {
		mt_unlock();
		ERR_PRINT("e->lock == 0");
		return nullptr;
	}

	if ((int)e->pos >= pool_size) {
		mt_unlock();
		ERR_PRINT("e->pos<0 || e->pos>=pool_size");
		return nullptr;
	}
	void *ptr = &pool[e->pos];

	mt_unlock();

	return ptr;
}
void PoolAllocator::unlock(ID p_mem) {
	if (!needs_locking) {
		return;
	}
	mt_lock();
	Entry *e = get_entry(p_mem);
	if (!e) {
		mt_unlock();
		ERR_FAIL_COND(!e);
	}
	if (e->lock == 0) {
		mt_unlock();
		ERR_PRINT("e->lock == 0");
		return;
	}
	e->lock--;
	mt_unlock();
}

int PoolAllocator::get_used_mem() const {
	return pool_size - free_mem;
}

int PoolAllocator::get_free_peak() {
	return free_mem_peak;
}

int PoolAllocator::get_free_mem() {
	return free_mem;
}

void PoolAllocator::create_pool(void *p_mem, int p_size, int p_max_entries) {
	pool = (uint8_t *)p_mem;
	pool_size = p_size;

	entry_array = memnew_arr(Entry, p_max_entries);
	entry_indices = memnew_arr(int, p_max_entries);
	entry_max = p_max_entries;
	entry_count = 0;

	free_mem = p_size;
	free_mem_peak = p_size;

	check_count = 0;
}

PoolAllocator::PoolAllocator(int p_size, bool p_needs_locking, int p_max_entries) {
	mem_ptr = memalloc(p_size);
	ERR_FAIL_COND(!mem_ptr);
	align = 1;
	create_pool(mem_ptr, p_size, p_max_entries);
	needs_locking = p_needs_locking;
}

PoolAllocator::PoolAllocator(void *p_mem, int p_size, int p_align, bool p_needs_locking, int p_max_entries) {
	if (p_align > 1) {
		uint8_t *mem8 = (uint8_t *)p_mem;
		uint64_t ofs = (uint64_t)mem8;
		if (ofs % p_align) {
			int dif = p_align - (ofs % p_align);
			mem8 += p_align - (ofs % p_align);
			p_size -= dif;
			p_mem = (void *)mem8;
		};
	};

	create_pool(p_mem, p_size, p_max_entries);
	needs_locking = p_needs_locking;
	align = p_align;
	mem_ptr = nullptr;
}

PoolAllocator::PoolAllocator(int p_align, int p_size, bool p_needs_locking, int p_max_entries) {
	ERR_FAIL_COND(p_align < 1);
	mem_ptr = Memory::alloc_static(p_size + p_align, true);
	uint8_t *mem8 = (uint8_t *)mem_ptr;
	uint64_t ofs = (uint64_t)mem8;
	if (ofs % p_align) {
		mem8 += p_align - (ofs % p_align);
	}
	create_pool(mem8, p_size, p_max_entries);
	needs_locking = p_needs_locking;
	align = p_align;
}

PoolAllocator::~PoolAllocator() {
	if (mem_ptr) {
		memfree(mem_ptr);
	}

	memdelete_arr(entry_array);
	memdelete_arr(entry_indices);
}

#undef COMPACT_CHUNK
#line 0

#line 1 "sfw/core/random_pcg.cpp"
/*************************************************************************/
/*  random_pcg.cpp                                                       */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

RandomPCG::RandomPCG(uint64_t p_seed, uint64_t p_inc) :
		pcg(),
		current_inc(p_inc) {
	seed(p_seed);
}

void RandomPCG::randomize() {
	seed((SFWTime::date() + SFWTime::time_us()) * pcg.state + PCG_DEFAULT_INC_64);
}

double RandomPCG::random(double p_from, double p_to) {
	return randd() * (p_to - p_from) + p_from;
}

float RandomPCG::random(float p_from, float p_to) {
	return randf() * (p_to - p_from) + p_from;
}

int RandomPCG::random(int p_from, int p_to) {
	if (p_from == p_to) {
		return p_from;
	}
	return rand(abs(p_from - p_to) + 1) + MIN(p_from, p_to);
}
#line 0

#line 1 "sfw/core/math_funcs.cpp"
/*************************************************************************/
/*  math_funcs.cpp                                                       */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

RandomPCG Math::default_rand(RandomPCG::DEFAULT_SEED, RandomPCG::DEFAULT_INC);

#define PHI 0x9e3779b9

uint32_t Math::rand_from_seed(uint64_t *seed) {
	RandomPCG rng = RandomPCG(*seed, RandomPCG::DEFAULT_INC);
	uint32_t r = rng.rand();
	*seed = rng.get_seed();
	return r;
}

void Math::seed(uint64_t x) {
	default_rand.seed(x);
}

void Math::randomize() {
	default_rand.randomize();
}

uint32_t Math::rand() {
	return default_rand.rand();
}

double Math::randfn(double mean, double deviation) {
	return default_rand.randfn(mean, deviation);
}

int Math::step_decimals(double p_step) {
	static const int maxn = 10;
	static const double sd[maxn] = {
		0.9999, // somehow compensate for floating point error
		0.09999,
		0.009999,
		0.0009999,
		0.00009999,
		0.000009999,
		0.0000009999,
		0.00000009999,
		0.000000009999,
		0.0000000009999
	};

	double abs = Math::abs(p_step);
	double decs = abs - (int)abs; // Strip away integer part
	for (int i = 0; i < maxn; i++) {
		if (decs >= sd[i]) {
			return i;
		}
	}

	return 0;
}

// Only meant for editor usage in float ranges, where a step of 0
// means that decimal digits should not be limited in String::num.
int Math::range_step_decimals(double p_step) {
	if (p_step < 0.0000000000001) {
		return 16; // Max value hardcoded in String::num
	}
	return step_decimals(p_step);
}

double Math::dectime(double p_value, double p_amount, double p_step) {
	double sgn = p_value < 0 ? -1.0 : 1.0;
	double val = Math::abs(p_value);
	val -= p_amount * p_step;
	if (val < 0.0) {
		val = 0.0;
	}
	return val * sgn;
}

double Math::ease(double p_x, double p_c) {
	if (p_x < 0) {
		p_x = 0;
	} else if (p_x > 1.0) {
		p_x = 1.0;
	}
	if (p_c > 0) {
		if (p_c < 1.0) {
			return 1.0 - Math::pow(1.0 - p_x, 1.0 / p_c);
		} else {
			return Math::pow(p_x, p_c);
		}
	} else if (p_c < 0) {
		//inout ease

		if (p_x < 0.5) {
			return Math::pow(p_x * 2.0, -p_c) * 0.5;
		} else {
			return (1.0 - Math::pow(1.0 - (p_x - 0.5) * 2.0, -p_c)) * 0.5 + 0.5;
		}
	} else {
		return 0; // no ease (raw)
	}
}

double Math::stepify(double p_value, double p_step) {
	if (p_step != 0) {
		p_value = Math::floor(p_value / p_step + 0.5) * p_step;
	}
	return p_value;
}

uint32_t Math::larger_prime(uint32_t p_val) {
	static const uint32_t primes[] = {
		5,
		13,
		23,
		47,
		97,
		193,
		389,
		769,
		1543,
		3079,
		6151,
		12289,
		24593,
		49157,
		98317,
		196613,
		393241,
		786433,
		1572869,
		3145739,
		6291469,
		12582917,
		25165843,
		50331653,
		100663319,
		201326611,
		402653189,
		805306457,
		1610612741,
		0,
	};

	int idx = 0;
	while (true) {
		ERR_FAIL_COND_V(primes[idx] == 0, 0);
		if (primes[idx] > p_val) {
			return primes[idx];
		}
		idx++;
	}
}

double Math::random(double from, double to) {
	return default_rand.random(from, to);
}

float Math::random(float from, float to) {
	return default_rand.random(from, to);
}

real_t Math::randomr(real_t from, real_t to) {
	return default_rand.randomr(from, to);
}

int Math::random(int from, int to) {
	return default_rand.random(from, to);
}
#line 0

#line 1 "sfw/core/ustring.cpp"
/*************************************************************************/
/*  ustring.cpp                                                          */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS // to disable build-time warning which suggested to use strcpy_s instead strcpy
#endif

#if defined(MINGW_ENABLED) || defined(_MSC_VER)
#define snprintf _snprintf_s
#endif

#define MAX_DIGITS 6
#define UPPERCASE(m_c) (((m_c) >= 'a' && (m_c) <= 'z') ? ((m_c) - ('a' - 'A')) : (m_c))
#define LOWERCASE(m_c) (((m_c) >= 'A' && (m_c) <= 'Z') ? ((m_c) + ('a' - 'A')) : (m_c))
#define IS_DIGIT(m_d) ((m_d) >= '0' && (m_d) <= '9')
#define IS_HEX_DIGIT(m_d) (((m_d) >= '0' && (m_d) <= '9') || ((m_d) >= 'a' && (m_d) <= 'f') || ((m_d) >= 'A' && (m_d) <= 'F'))

#define READING_SIGN 0
#define READING_INT 1
#define READING_DEC 2
#define READING_EXP 3
#define READING_DONE 4

const char CharString::_null = 0;
const char16_t Char16String::_null = 0;
const CharType String::_null = 0;

static bool _wildcard_match(const CharType *p_pattern, const CharType *p_string, bool p_case_sensitive) {
	switch (*p_pattern) {
		case '\0':
			return !*p_string;
		case '*':
			return _wildcard_match(p_pattern + 1, p_string, p_case_sensitive) || (*p_string && _wildcard_match(p_pattern, p_string + 1, p_case_sensitive));
		case '?':
			return *p_string && (*p_string != '.') && _wildcard_match(p_pattern + 1, p_string + 1, p_case_sensitive);
		default:

			return (p_case_sensitive ? (*p_string == *p_pattern) : (_find_upper(*p_string) == _find_upper(*p_pattern))) && _wildcard_match(p_pattern + 1, p_string + 1, p_case_sensitive);
	}
}

static int _humanize_digits(int p_num) {
	if (p_num < 100) {
		return 2;
	} else if (p_num < 1024) {
		return 1;
	} else {
		return 0;
	}
}

bool select_word(const String &p_s, int p_col, int &r_beg, int &r_end) {
	const String &s = p_s;
	int beg = CLAMP(p_col, 0, s.length());
	int end = beg;

	if (s[beg] > 32 || beg == s.length()) {
		bool symbol = beg < s.length() && is_symbol(s[beg]);

		while (beg > 0 && s[beg - 1] > 32 && (symbol == is_symbol(s[beg - 1]))) {
			beg--;
		}
		while (end < s.length() && s[end + 1] > 32 && (symbol == is_symbol(s[end + 1]))) {
			end++;
		}

		if (end < s.length()) {
			end += 1;
		}

		r_beg = beg;
		r_end = end;

		return true;
	} else {
		return false;
	}
}

/*************************************************************************/
/*  Char16String                                                         */
/*************************************************************************/

bool Char16String::operator<(const Char16String &p_right) const {
	if (length() == 0) {
		return p_right.length() != 0;
	}

	return is_str_less(get_data(), p_right.get_data());
}

Char16String &Char16String::operator+=(char16_t p_char) {
	const int lhs_len = length();
	resize(lhs_len + 2);

	char16_t *dst = ptrw();
	dst[lhs_len] = p_char;
	dst[lhs_len + 1] = 0;

	return *this;
}

void Char16String::operator=(const char16_t *p_cstr) {
	copy_from(p_cstr);
}

const char16_t *Char16String::get_data() const {
	if (size()) {
		return &operator[](0);
	} else {
		return u"";
	}
}

void Char16String::copy_from(const char16_t *p_cstr) {
	if (!p_cstr) {
		resize(0);
		return;
	}

	const char16_t *s = p_cstr;
	for (; *s; s++) {
	}
	size_t len = s - p_cstr;

	if (len == 0) {
		resize(0);
		return;
	}

	Error err = resize(++len); // include terminating null char

	ERR_FAIL_COND_MSG(err != OK, "Failed to copy char16_t string.");

	memcpy(ptrw(), p_cstr, len * sizeof(char16_t));
}

/*************************************************************************/
/*  CharString                                                           */
/*************************************************************************/

bool CharString::operator<(const CharString &p_right) const {
	if (length() == 0) {
		return p_right.length() != 0;
	}

	return is_str_less(get_data(), p_right.get_data());
}

CharString &CharString::operator+=(char p_char) {
	const int lhs_len = length();
	resize(lhs_len + 2);

	char *dst = ptrw();
	dst[lhs_len] = p_char;
	dst[lhs_len + 1] = 0;

	return *this;
}

void CharString::operator=(const char *p_cstr) {
	copy_from(p_cstr);
}

const char *CharString::get_data() const {
	if (size()) {
		return &operator[](0);
	} else {
		return "";
	}
}

void CharString::copy_from(const char *p_cstr) {
	if (!p_cstr) {
		resize(0);
		return;
	}

	size_t len = strlen(p_cstr);

	if (len == 0) {
		resize(0);
		return;
	}

	Error err = resize(++len); // include terminating null char

	ERR_FAIL_COND_MSG(err != OK, "Failed to copy C-string.");

	memcpy(ptrw(), p_cstr, len);
}

/*************************************************************************/
/*  String                                                               */
/*************************************************************************/

void String::copy_from(const char *p_cstr) {
	// copy Latin-1 encoded c-string directly
	if (!p_cstr) {
		resize(0);
		return;
	}

	const size_t len = strlen(p_cstr);

	if (len == 0) {
		resize(0);
		return;
	}

	resize(len + 1); // include 0

	CharType *dst = this->ptrw();

	for (size_t i = 0; i <= len; i++) {
		uint8_t c = p_cstr[i] >= 0 ? p_cstr[i] : uint8_t(256 + p_cstr[i]);
		if (c == 0 && i < len) {
			print_unicode_error("NUL character", true);
			dst[i] = 0x20;
		} else {
			dst[i] = c;
		}
	}
}

void String::copy_from(const char *p_cstr, const int p_clip_to) {
	// copy Latin-1 encoded c-string directly
	if (!p_cstr) {
		resize(0);
		return;
	}

	int len = 0;
	const char *ptr = p_cstr;
	while ((p_clip_to < 0 || len < p_clip_to) && *(ptr++) != 0) {
		len++;
	}

	if (len == 0) {
		resize(0);
		return;
	}

	resize(len + 1); // include 0

	CharType *dst = this->ptrw();

	for (int i = 0; i < len; i++) {
		uint8_t c = p_cstr[i] >= 0 ? p_cstr[i] : uint8_t(256 + p_cstr[i]);
		if (c == 0) {
			print_unicode_error("NUL character", true);
			dst[i] = 0x20;
		} else {
			dst[i] = c;
		}
	}
	dst[len] = 0;
}

void String::copy_from(const wchar_t *p_cstr) {
#ifdef WINDOWS_ENABLED
	// wchar_t is 16-bit, parse as UTF-16
	parse_utf16((const char16_t *)p_cstr);
#else
	// wchar_t is 32-bit, copy directly
	copy_from((const CharType *)p_cstr);
#endif
}

void String::copy_from(const wchar_t *p_cstr, const int p_clip_to) {
#ifdef WINDOWS_ENABLED
	// wchar_t is 16-bit, parse as UTF-16
	parse_utf16((const char16_t *)p_cstr, p_clip_to);
#else
	// wchar_t is 32-bit, copy directly
	copy_from((const CharType *)p_cstr, p_clip_to);
#endif
}

void String::copy_from(const CharType &p_char) {
	if (p_char == 0) {
		print_unicode_error("NUL character", true);
		return;
	}
	/*
	if ((p_char & 0xfffff800) == 0xd800) {
		print_unicode_error(vformat("Unpaired surrogate (%x)", (uint32_t)p_char));
	}
	if (p_char > 0x10ffff) {
		print_unicode_error(vformat("Invalid unicode codepoint (%x)", (uint32_t)p_char));
	}
	*/

	resize(2);

	CharType *dst = ptrw();
	dst[0] = p_char;
	dst[1] = 0;
}

void String::copy_from(const CharType *p_cstr) {
	if (!p_cstr) {
		resize(0);
		return;
	}

	int len = 0;
	const CharType *ptr = p_cstr;
	while (*(ptr++) != 0) {
		len++;
	}

	if (len == 0) {
		resize(0);
		return;
	}

	copy_from_unchecked(p_cstr, len);
}

void String::copy_from(const CharType *p_cstr, const int p_clip_to) {
	if (!p_cstr) {
		resize(0);
		return;
	}

	int len = 0;
	const CharType *ptr = p_cstr;
	while ((p_clip_to < 0 || len < p_clip_to) && *(ptr++) != 0) {
		len++;
	}

	if (len == 0) {
		resize(0);
		return;
	}

	copy_from_unchecked(p_cstr, len);
}

// assumes the following have already been validated:
// p_char != nullptr
// p_length > 0
// p_length <= p_char strlen
void String::copy_from_unchecked(const CharType *p_char, const int p_length) {
	resize(p_length + 1);
	CharType *dst = ptrw();
	dst[p_length] = 0;

	for (int i = 0; i < p_length; i++) {
		if (p_char[i] == 0) {
			print_unicode_error("NUL character", true);
			dst[i] = 0x20;
			continue;
		}
		/*
		if ((p_char[i] & 0xfffff800) == 0xd800) {
			print_unicode_error(vformat("Unpaired surrogate (%x)", (uint32_t)p_char[i]));
		}
		if (p_char[i] > 0x10ffff) {
			print_unicode_error(vformat("Invalid unicode codepoint (%x)", (uint32_t)p_char[i]));
		}
		*/
		dst[i] = p_char[i];
	}
}

bool String::operator==(const String &p_str) const {
	if (length() != p_str.length()) {
		return false;
	}
	if (empty()) {
		return true;
	}

	int l = length();

	const CharType *src = get_data();
	const CharType *dst = p_str.get_data();

	/* Compare char by char */
	for (int i = 0; i < l; i++) {
		if (src[i] != dst[i]) {
			return false;
		}
	}

	return true;
}

bool String::operator!=(const String &p_str) const {
	return !((*this == p_str));
}

String String::operator+(const String &p_str) const {
	String res = *this;
	res += p_str;
	return res;
}

/*
String String::operator+(CharType p_char) const {
	String res = *this;
	res += p_char;
	return res;
}
*/

String &String::operator+=(const String &p_str) {
	const int lhs_len = length();
	if (lhs_len == 0) {
		*this = p_str;
		return *this;
	}

	const int rhs_len = p_str.length();
	if (rhs_len == 0) {
		return *this;
	}

	set_length(lhs_len + rhs_len);

	const CharType *src = p_str.get_data();
	CharType *dst = ptrw() + lhs_len;

	memcpy(dst, src, rhs_len * sizeof(CharType));

	return *this;
}

String &String::operator+=(CharType p_char) {
	if (p_char == 0) {
		print_unicode_error("NUL character", true);
		return *this;
	}
	/*
	if ((p_char & 0xfffff800) == 0xd800) {
		print_unicode_error(vformat("Unpaired surrogate (%x)", (uint32_t)p_char));
	}
	if (p_char > 0x10ffff) {
		print_unicode_error(vformat("Invalid unicode codepoint (%x)", (uint32_t)p_char));
	}
	*/

	const int lhs_len = length();
	set_length(lhs_len + 1);
	CharType *dst = ptrw();
	dst[lhs_len] = p_char;

	return *this;
}

String &String::operator+=(const char *p_str) {
	if (!p_str || p_str[0] == 0) {
		return *this;
	}

	const int lhs_len = length();
	const size_t rhs_len = strlen(p_str);

	set_length(lhs_len + rhs_len);

	CharType *dst = ptrw() + lhs_len;

	for (size_t i = 0; i <= rhs_len; i++) {
		uint8_t c = p_str[i] >= 0 ? p_str[i] : uint8_t(256 + p_str[i]);
		if (c == 0 && i < rhs_len) {
			print_unicode_error("NUL character", true);
			dst[i] = 0x20;
		} else {
			dst[i] = c;
		}
	}

	return *this;
}

String &String::operator+=(const wchar_t *p_str) {
#ifdef WINDOWS_ENABLED
	// wchar_t is 16-bit
	*this += String::utf16((const char16_t *)p_str);
#else
	// wchar_t is 32-bit
	*this += String((const CharType *)p_str);
#endif
	return *this;
}

String &String::operator+=(const CharType *p_str) {
	*this += String(p_str);
	return *this;
}

void String::operator=(const char *p_str) {
	copy_from(p_str);
}

void String::operator=(const CharType *p_str) {
	copy_from(p_str);
}

void String::operator=(const wchar_t *p_str) {
	copy_from(p_str);
}

bool String::operator==(const char *p_str) const {
	// compare Latin-1 encoded c-string
	int len = 0;
	const char *aux = p_str;

	while (*(aux++) != 0) {
		len++;
	}

	if (length() != len) {
		return false;
	}
	if (empty()) {
		return true;
	}

	int l = length();

	const CharType *dst = get_data();

	// Compare char by char
	for (int i = 0; i < l; i++) {
		if ((CharType)p_str[i] != dst[i]) {
			return false;
		}
	}

	return true;
}

bool String::operator==(const wchar_t *p_str) const {
#ifdef WINDOWS_ENABLED
	// wchar_t is 16-bit, parse as UTF-16
	return *this == String::utf16((const char16_t *)p_str);
#else
	// wchar_t is 32-bit, compare char by char
	return *this == (const CharType *)p_str;
#endif
}

bool String::operator==(const CharType *p_str) const {
	int len = 0;
	const CharType *aux = p_str;

	while (*(aux++) != 0) {
		len++;
	}

	if (length() != len) {
		return false;
	}
	if (empty()) {
		return true;
	}

	int l = length();

	const CharType *dst = get_data();

	/* Compare char by char */
	for (int i = 0; i < l; i++) {
		if (p_str[i] != dst[i]) {
			return false;
		}
	}

	return true;
}

bool String::operator==(const StrRange &p_str_range) const {
	int len = p_str_range.len;

	if (length() != len) {
		return false;
	}
	if (empty()) {
		return true;
	}

	const CharType *c_str = p_str_range.c_str;
	const CharType *dst = &operator[](0);

	/* Compare char by char */
	for (int i = 0; i < len; i++) {
		if (c_str[i] != dst[i]) {
			return false;
		}
	}

	return true;
}

bool String::operator!=(const char *p_str) const {
	return (!(*this == p_str));
}

bool String::operator!=(const wchar_t *p_str) const {
	return (!(*this == p_str));
}

bool String::operator!=(const CharType *p_str) const {
	return (!(*this == p_str));
}

bool String::operator<(const char *p_str) const {
	if (empty() && p_str[0] == 0) {
		return false;
	}
	if (empty()) {
		return true;
	}
	return is_str_less(get_data(), p_str);
}

bool String::operator<(const wchar_t *p_str) const {
	if (empty() && p_str[0] == 0) {
		return false;
	}
	if (empty()) {
		return true;
	}

#ifdef WINDOWS_ENABLED
	// wchar_t is 16-bit
	return is_str_less(get_data(), String::utf16((const char16_t *)p_str).get_data());
#else
	// wchar_t is 32-bit
	return is_str_less(get_data(), (const CharType *)p_str);
#endif
}

bool String::operator<(const CharType *p_str) const {
	if (empty() && p_str[0] == 0) {
		return false;
	}
	if (empty()) {
		return true;
	}

	return is_str_less(get_data(), p_str);
}

bool String::operator<(const String &p_str) const {
	return operator<(p_str.get_data());
}

bool String::operator<=(const String &p_str) const {
	return !(p_str < *this);
}

bool String::operator>(const String &p_str) const {
	return p_str < *this;
}

bool String::operator>=(const String &p_str) const {
	return !(*this < p_str);
}

signed char String::nocasecmp_to(const String &p_str) const {
	if (empty() && p_str.empty()) {
		return 0;
	}
	if (empty()) {
		return -1;
	}
	if (p_str.empty()) {
		return 1;
	}

	const CharType *that_str = p_str.get_data();
	const CharType *this_str = get_data();

	while (true) {
		if (*that_str == 0 && *this_str == 0) {
			return 0; //we're equal
		} else if (*this_str == 0) {
			return -1; //if this is empty, and the other one is not, then we're less.. I think?
		} else if (*that_str == 0) {
			return 1; //otherwise the other one is smaller..
		} else if (_find_upper(*this_str) < _find_upper(*that_str)) { //more than
			return -1;
		} else if (_find_upper(*this_str) > _find_upper(*that_str)) { //less than
			return 1;
		}

		this_str++;
		that_str++;
	}
}

signed char String::casecmp_to(const String &p_str) const {
	if (empty() && p_str.empty()) {
		return 0;
	}
	if (empty()) {
		return -1;
	}
	if (p_str.empty()) {
		return 1;
	}

	const CharType *that_str = p_str.get_data();
	const CharType *this_str = get_data();

	while (true) {
		if (*that_str == 0 && *this_str == 0) {
			return 0; //we're equal
		} else if (*this_str == 0) {
			return -1; //if this is empty, and the other one is not, then we're less.. I think?
		} else if (*that_str == 0) {
			return 1; //otherwise the other one is smaller..
		} else if (*this_str < *that_str) { //more than
			return -1;
		} else if (*this_str > *that_str) { //less than
			return 1;
		}

		this_str++;
		that_str++;
	}
}

signed char String::naturalnocasecmp_to(const String &p_str) const {
	const CharType *this_str = get_data();
	const CharType *that_str = p_str.get_data();

	if (this_str && that_str) {
		while (*this_str == '.' || *that_str == '.') {
			if (*this_str++ != '.') {
				return 1;
			}
			if (*that_str++ != '.') {
				return -1;
			}
			if (!*that_str) {
				return 1;
			}
			if (!*this_str) {
				return -1;
			}
		}

		while (*this_str) {
			if (!*that_str) {
				return 1;
			} else if (is_digit(*this_str)) {
				if (!is_digit(*that_str)) {
					return -1;
				}

				// Keep ptrs to start of numerical sequences
				const CharType *this_substr = this_str;
				const CharType *that_substr = that_str;

				// Compare lengths of both numerical sequences, ignoring leading zeros
				while (is_digit(*this_str)) {
					this_str++;
				}
				while (is_digit(*that_str)) {
					that_str++;
				}
				while (*this_substr == '0') {
					this_substr++;
				}
				while (*that_substr == '0') {
					that_substr++;
				}
				int this_len = this_str - this_substr;
				int that_len = that_str - that_substr;

				if (this_len < that_len) {
					return -1;
				} else if (this_len > that_len) {
					return 1;
				}

				// If lengths equal, compare lexicographically
				while (this_substr != this_str && that_substr != that_str) {
					if (*this_substr < *that_substr) {
						return -1;
					} else if (*this_substr > *that_substr) {
						return 1;
					}
					this_substr++;
					that_substr++;
				}
			} else if (is_digit(*that_str)) {
				return 1;
			} else {
				if (_find_upper(*this_str) < _find_upper(*that_str)) { //more than
					return -1;
				} else if (_find_upper(*this_str) > _find_upper(*that_str)) { //less than
					return 1;
				}

				this_str++;
				that_str++;
			}
		}
		if (*that_str) {
			return -1;
		}
	}

	return 0;
}

const CharType *String::get_data() const {
	static const CharType zero = 0;
	return size() ? &operator[](0) : &zero;
}

bool String::is_valid_string() const {
	int l = length();
	const CharType *src = get_data();
	bool valid = true;
	for (int i = 0; i < l; i++) {
		valid = valid && (src[i] < 0xd800 || (src[i] > 0xdfff && src[i] <= 0x10ffff));
	}
	return valid;
}

void String::print_unicode_error(const String &p_message, bool p_critical) const {
	/*
	if (p_critical) {
		print_error(vformat("Unicode parsing error, some characters were replaced with spaces: %s", p_message));
	} else {
		print_error(vformat("Unicode parsing error: %s", p_message));
	}
	*/
}

/* complex helpers */

String String::substr(int p_from, int p_chars) const {
	if (p_chars == -1) {
		p_chars = length() - p_from;
	}

	if (empty() || p_from < 0 || p_from >= length() || p_chars <= 0) {
		return "";
	}

	if ((p_from + p_chars) > length()) {
		p_chars = length() - p_from;
	}

	if (p_from == 0 && p_chars >= length()) {
		return String(*this);
	}

	String s = String();
	s.copy_from_unchecked(&get_data()[p_from], p_chars);
	return s;
}

String String::substr_index(const int start_index, const int end_index) const {
	int s = length();

	if (start_index < 0 || start_index >= s || end_index < 0) {
		return "";
	}

	if (start_index > end_index) {
		return "";
	}

	if (end_index >= s) {
		return substr(start_index, (s - 1) - start_index);
	}

	return substr(start_index, end_index - start_index);
}

int String::find(const String &p_str, int p_from) const {
	if (p_from < 0) {
		return -1;
	}

	const int src_len = p_str.length();

	const int len = length();

	if (src_len == 0 || len == 0) {
		return -1; // won't find anything!
	}

	const CharType *src = get_data();
	const CharType *str = p_str.get_data();

	for (int i = p_from; i <= (len - src_len); i++) {
		bool found = true;
		for (int j = 0; j < src_len; j++) {
			int read_pos = i + j;

			if (read_pos >= len) {
				ERR_PRINT("read_pos>=len");
				return -1;
			};

			if (src[read_pos] != str[j]) {
				found = false;
				break;
			}
		}

		if (found) {
			return i;
		}
	}

	return -1;
}

int String::find(const char *p_str, int p_from) const {
	if (p_from < 0) {
		return -1;
	}

	const int len = length();

	if (len == 0) {
		return -1; // won't find anything!
	}

	const CharType *src = get_data();

	int src_len = 0;
	while (p_str[src_len] != '\0') {
		src_len++;
	}

	if (src_len == 1) {
		const CharType needle = p_str[0];

		for (int i = p_from; i < len; i++) {
			if (src[i] == needle) {
				return i;
			}
		}

	} else {
		for (int i = p_from; i <= (len - src_len); i++) {
			bool found = true;
			for (int j = 0; j < src_len; j++) {
				int read_pos = i + j;

				if (read_pos >= len) {
					ERR_PRINT("read_pos>=len");
					return -1;
				};

				if (src[read_pos] != (CharType)p_str[j]) {
					found = false;
					break;
				}
			}

			if (found) {
				return i;
			}
		}
	}

	return -1;
}

int String::find_char(const CharType &p_char, int p_from) const {
	return _cowdata.find(p_char, p_from);
}

int String::find_last(const String &p_str) const {
	return rfind(p_str);
}

int String::findn(const String &p_str, int p_from) const {
	if (p_from < 0) {
		return -1;
	}

	int src_len = p_str.length();

	if (src_len == 0 || length() == 0) {
		return -1; // won't find anything!
	}

	const CharType *srcd = get_data();

	for (int i = p_from; i <= (length() - src_len); i++) {
		bool found = true;
		for (int j = 0; j < src_len; j++) {
			int read_pos = i + j;

			if (read_pos >= length()) {
				ERR_PRINT("read_pos>=length()");
				return -1;
			};

			CharType src = _find_lower(srcd[read_pos]);
			CharType dst = _find_lower(p_str[j]);

			if (src != dst) {
				found = false;
				break;
			}
		}

		if (found) {
			return i;
		}
	}

	return -1;
}

int String::rfind(const String &p_str, int p_from) const {
	// establish a limit
	int limit = length() - p_str.length();
	if (limit < 0) {
		return -1;
	}

	// establish a starting point
	if (p_from < 0) {
		p_from = limit;
	} else if (p_from > limit) {
		p_from = limit;
	}

	int src_len = p_str.length();
	int len = length();

	if (src_len == 0 || len == 0) {
		return -1; // won't find anything!
	}

	const CharType *src = get_data();

	for (int i = p_from; i >= 0; i--) {
		bool found = true;
		for (int j = 0; j < src_len; j++) {
			int read_pos = i + j;

			if (read_pos >= len) {
				ERR_PRINT("read_pos>=len");
				return -1;
			};

			if (src[read_pos] != p_str[j]) {
				found = false;
				break;
			}
		}

		if (found) {
			return i;
		}
	}

	return -1;
}

int String::rfindn(const String &p_str, int p_from) const {
	// establish a limit
	int limit = length() - p_str.length();
	if (limit < 0) {
		return -1;
	}

	// establish a starting point
	if (p_from < 0) {
		p_from = limit;
	} else if (p_from > limit) {
		p_from = limit;
	}

	int src_len = p_str.length();
	int len = length();

	if (src_len == 0 || len == 0) {
		return -1; // won't find anything!
	}

	const CharType *src = get_data();

	for (int i = p_from; i >= 0; i--) {
		bool found = true;
		for (int j = 0; j < src_len; j++) {
			int read_pos = i + j;

			if (read_pos >= len) {
				ERR_PRINT("read_pos>=len");
				return -1;
			};

			CharType srcc = _find_lower(src[read_pos]);
			CharType dstc = _find_lower(p_str[j]);

			if (srcc != dstc) {
				found = false;
				break;
			}
		}

		if (found) {
			return i;
		}
	}

	return -1;
}

int String::findmk(const Vector<String> &p_keys, int p_from, int *r_key) const {
	if (p_from < 0) {
		return -1;
	}
	if (p_keys.size() == 0) {
		return -1;
	}

	//int src_len=p_str.length();
	const String *keys = &p_keys[0];
	int key_count = p_keys.size();
	int len = length();

	if (len == 0) {
		return -1; // won't find anything!
	}

	const CharType *src = get_data();

	for (int i = p_from; i < len; i++) {
		bool found = true;
		for (int k = 0; k < key_count; k++) {
			found = true;
			if (r_key) {
				*r_key = k;
			}
			const CharType *cmp = keys[k].get_data();
			int l = keys[k].length();

			for (int j = 0; j < l; j++) {
				int read_pos = i + j;

				if (read_pos >= len) {
					found = false;
					break;
				};

				if (src[read_pos] != cmp[j]) {
					found = false;
					break;
				}
			}
			if (found) {
				break;
			}
		}

		if (found) {
			return i;
		}
	}

	return -1;
}

int String::find_first_difference_index(const String &p_str) const {
	const int olen = p_str.length();
	const int len = length();
	const int c = len < olen ? len : olen;

	const CharType *p = get_data();
	const CharType *op = p_str.get_data();

	for (int i = 0; i < c; ++i) {
		if (p[i] != op[i]) {
			return i;
		}
	}

	return c;
}

bool String::is_word_at(const int index, const char *p_str) const {
	int size = length();

	ERR_FAIL_INDEX_V(index, size, false);

	int i = 0;

	while (p_str[i] != '\0') {
		int iind = index + i;

		if (iind >= size) {
			return false;
		}

		if (operator[](iind) != (CharType)p_str[i]) {
			return false;
		}

		++i;
	}

	return true;
}
bool String::is_word_at(const int index, const String &p_str) const {
	int size = length();

	ERR_FAIL_INDEX_V(index, size, false);

	if (index + p_str.length() >= size) {
		return false;
	}

	for (int i = 0; i < p_str.length(); ++i) {
		int iind = index + i;

		if (operator[](iind) != p_str[i]) {
			return false;
		}
	}

	return true;
}

bool String::match(const String &p_wildcard) const {
	if (!p_wildcard.length() || !length()) {
		return false;
	}

	return _wildcard_match(p_wildcard.get_data(), get_data(), true);
}

bool String::matchn(const String &p_wildcard) const {
	if (!p_wildcard.length() || !length()) {
		return false;
	}
	return _wildcard_match(p_wildcard.get_data(), get_data(), false);
}

bool String::begins_with(const String &p_string) const {
	int l = p_string.length();
	if (l > length()) {
		return false;
	}

	if (l == 0) {
		return true;
	}

	const CharType *p = &p_string[0];
	const CharType *s = &operator[](0);

	for (int i = 0; i < l; i++) {
		if (p[i] != s[i]) {
			return false;
		}
	}

	return true;
}

bool String::begins_with(const char *p_string) const {
	int l = length();
	if (l == 0 || !p_string) {
		return false;
	}

	const CharType *str = &operator[](0);
	int i = 0;

	while (*p_string && i < l) {
		if ((CharType)(*p_string) != str[i]) {
			return false;
		}
		i++;
		p_string++;
	}

	return *p_string == 0;
}

bool String::ends_with(const String &p_string) const {
	int l = p_string.length();
	if (l > length()) {
		return false;
	}

	if (l == 0) {
		return true;
	}

	const CharType *p = &p_string[0];
	const CharType *s = &operator[](length() - l);

	for (int i = 0; i < l; i++) {
		if (p[i] != s[i]) {
			return false;
		}
	}

	return true;
}

bool String::is_enclosed_in(const String &p_string) const {
	return begins_with(p_string) && ends_with(p_string);
}

bool String::is_subsequence_of(const String &p_string) const {
	return _base_is_subsequence_of(p_string, false);
}

bool String::is_subsequence_ofi(const String &p_string) const {
	return _base_is_subsequence_of(p_string, true);
}

bool String::is_quoted() const {
	return is_enclosed_in("\"") || is_enclosed_in("'");
}

Vector<String> String::bigrams() const {
	int n_pairs = length() - 1;
	Vector<String> b;
	if (n_pairs <= 0) {
		return b;
	}
	b.resize(n_pairs);
	for (int i = 0; i < n_pairs; i++) {
		b.write[i] = substr(i, 2);
	}
	return b;
}

// Similarity according to Sorensen-Dice coefficient
float String::similarity(const String &p_string) const {
	if (operator==(p_string)) {
		// Equal strings are totally similar
		return 1.0f;
	}
	if (length() < 2 || p_string.length() < 2) {
		// No way to calculate similarity without a single bigram
		return 0.0f;
	}

	Vector<String> src_bigrams = bigrams();
	Vector<String> tgt_bigrams = p_string.bigrams();

	int src_size = src_bigrams.size();
	int tgt_size = tgt_bigrams.size();

	float sum = src_size + tgt_size;
	float inter = 0;
	for (int i = 0; i < src_size; i++) {
		for (int j = 0; j < tgt_size; j++) {
			if (src_bigrams[i] == tgt_bigrams[j]) {
				inter++;
				break;
			}
		}
	}

	return (2.0f * inter) / sum;
}

/*
String String::format(const Variant &values, String placeholder) const {
	String new_string = String(this->ptr());

	if (values.get_type() == Variant::ARRAY) {
		Array values_arr = values;

		for (int i = 0; i < values_arr.size(); i++) {
			String i_as_str = String::num_int64(i);

			if (values_arr[i].get_type() == Variant::ARRAY) { //Array in Array structure [["name","RobotGuy"],[0,"pandemonium"],["strength",9000.91]]
				Array value_arr = values_arr[i];

				if (value_arr.size() == 2) {
					Variant v_key = value_arr[0];
					String key = v_key;

					Variant v_val = value_arr[1];
					String val = v_val;

					new_string = new_string.replace(placeholder.replace("_", key), val);
				} else {
					ERR_PRINT(String("STRING.format Inner Array size != 2 ").ascii().get_data());
				}
			} else { //Array structure ["RobotGuy","Logis","rookie"]
				Variant v_val = values_arr[i];
				String val = v_val;

				if (placeholder.find("_") > -1) {
					new_string = new_string.replace(placeholder.replace("_", i_as_str), val);
				} else {
					new_string = new_string.replace_first(placeholder, val);
				}
			}
		}
	} else if (values.get_type() == Variant::DICTIONARY) {
		Dictionary d = values;
		List<Variant> keys;
		d.get_key_list(&keys);

		for (List<Variant>::Element *E = keys.front(); E; E = E->next()) {
			String key = E->get();
			String val = d[E->get()];

			new_string = new_string.replace(placeholder.replace("_", key), val);
		}
	} else {
		ERR_PRINT(String("Invalid type: use Array or Dictionary.").ascii().get_data());
	}

	return new_string;
}
*/

String String::replace_first(const String &p_key, const String &p_with) const {
	int pos = find(p_key);
	if (pos >= 0) {
		return substr(0, pos) + p_with + substr(pos + p_key.length(), length());
	}

	return *this;
}

String String::replace(const String &p_key, const String &p_with) const {
	String new_string;
	int search_from = 0;
	int result = 0;

	while ((result = find(p_key, search_from)) >= 0) {
		new_string += substr(search_from, result - search_from);
		new_string += p_with;
		search_from = result + p_key.length();
	}

	if (search_from == 0) {
		return *this;
	}

	new_string += substr(search_from, length() - search_from);

	return new_string;
}

String String::replace(const char *p_key, const char *p_with) const {
	String new_string;
	int search_from = 0;
	int result = 0;

	while ((result = find(p_key, search_from)) >= 0) {
		new_string += substr(search_from, result - search_from);
		new_string += p_with;
		int k = 0;
		while (p_key[k] != '\0') {
			k++;
		}
		search_from = result + k;
	}

	if (search_from == 0) {
		return *this;
	}

	new_string += substr(search_from, length() - search_from);

	return new_string;
}

String String::replacen(const String &p_key, const String &p_with) const {
	String new_string;
	int search_from = 0;
	int result = 0;

	while ((result = findn(p_key, search_from)) >= 0) {
		new_string += substr(search_from, result - search_from);
		new_string += p_with;
		search_from = result + p_key.length();
	}

	if (search_from == 0) {
		return *this;
	}

	new_string += substr(search_from, length() - search_from);
	return new_string;
}

String String::newline_to_br() const {
	String r = replace("\r\n", "<br>");
	return r.replace("\n", "<br>");
}

String String::repeat(int p_count) const {
	ERR_FAIL_COND_V_MSG(p_count < 0, "", "Parameter count should be a positive number.");

	int len = length();
	String new_string = *this;
	new_string.resize(p_count * len + 1);

	CharType *dst = new_string.ptrw();
	int offset = 1;
	int stride = 1;
	while (offset < p_count) {
		memcpy(dst + offset * len, dst, stride * len * sizeof(CharType));
		offset += stride;
		stride = MIN(stride * 2, p_count - offset);
	}
	dst[p_count * len] = _null;

	return new_string;
}

String String::insert(int p_at_pos, const String &p_string) const {
	if (p_at_pos < 0) {
		return *this;
	}

	if (p_at_pos > length()) {
		p_at_pos = length();
	}

	String pre;
	if (p_at_pos > 0) {
		pre = substr(0, p_at_pos);
	}

	String post;
	if (p_at_pos < length()) {
		post = substr(p_at_pos, length() - p_at_pos);
	}

	return pre + p_string + post;
}

String String::pad_decimals(int p_digits) const {
	String s = *this;
	int c = s.find(".");

	if (c == -1) {
		if (p_digits <= 0) {
			return s;
		}
		s += ".";
		c = s.length() - 1;
	} else {
		if (p_digits <= 0) {
			return s.substr(0, c);
		}
	}

	if (s.length() - (c + 1) > p_digits) {
		s = s.substr(0, c + p_digits + 1);
	} else {
		while (s.length() - (c + 1) < p_digits) {
			s += "0";
		}
	}
	return s;
}

String String::pad_zeros(int p_digits) const {
	String s = *this;
	int end = s.find(".");

	if (end == -1) {
		end = s.length();
	}

	if (end == 0) {
		return s;
	}

	int begin = 0;

	while (begin < end && (s[begin] < '0' || s[begin] > '9')) {
		begin++;
	}

	if (begin >= end) {
		return s;
	}

	while (end - begin < p_digits) {
		s = s.insert(begin, "0");
		end++;
	}

	return s;
}

String String::trim_prefix(const String &p_prefix) const {
	String s = *this;
	if (s.begins_with(p_prefix)) {
		return s.substr(p_prefix.length(), s.length() - p_prefix.length());
	}
	return s;
}

String String::trim_suffix(const String &p_suffix) const {
	String s = *this;
	if (s.ends_with(p_suffix)) {
		return s.substr(0, s.length() - p_suffix.length());
	}
	return s;
}

// Left-pad with a character.
String String::lpad(int min_length, const String &character) const {
	String s = *this;
	int padding = min_length - s.length();
	if (padding > 0) {
		for (int i = 0; i < padding; i++) {
			s = character + s;
		}
	}

	return s;
}

// Right-pad with a character.
String String::rpad(int min_length, const String &character) const {
	String s = *this;
	int padding = min_length - s.length();
	if (padding > 0) {
		for (int i = 0; i < padding; i++) {
			s = s + character;
		}
	}

	return s;
}

String String::quote(String quotechar) const {
	return quotechar + *this + quotechar;
}

String String::unquote() const {
	if (!is_quoted()) {
		return *this;
	}

	return substr(1, length() - 2);
}

String String::num(double p_num, int p_decimals) {
	if (Math::is_nan(p_num)) {
		return "nan";
	}

	if (Math::is_inf(p_num)) {
		if (signbit(p_num)) {
			return "-inf";
		} else {
			return "inf";
		}
	}

#ifndef NO_USE_STDLIB

	if (p_decimals > 16) {
		p_decimals = 16;
	}

	char fmt[7];
	fmt[0] = '%';
	fmt[1] = '.';

	if (p_decimals < 0) {
		fmt[1] = 'l';
		fmt[2] = 'f';
		fmt[3] = 0;

	} else if (p_decimals < 10) {
		fmt[2] = '0' + p_decimals;
		fmt[3] = 'l';
		fmt[4] = 'f';
		fmt[5] = 0;
	} else {
		fmt[2] = '0' + (p_decimals / 10);
		fmt[3] = '0' + (p_decimals % 10);
		fmt[4] = 'l';
		fmt[5] = 'f';
		fmt[6] = 0;
	}
	char buf[256];

#if defined(__GNUC__) || defined(_MSC_VER)
	snprintf(buf, 256, fmt, p_num);
#else
	sprintf(buf, fmt, p_num);
#endif

	buf[255] = 0;
	//destroy trailing zeroes
	{
		bool period = false;
		int z = 0;
		while (buf[z]) {
			if (buf[z] == '.') {
				period = true;
			}
			z++;
		}

		if (period) {
			z--;
			while (z > 0) {
				if (buf[z] == '0') {
					buf[z] = 0;
				} else if (buf[z] == '.') {
					buf[z] = 0;
					break;
				} else {
					break;
				}

				z--;
			}
		}
	}

	return buf;
#else

	String s;
	String sd;
	/* integer part */

	bool neg = p_num < 0;
	p_num = ABS(p_num);
	int intn = (int)p_num;

	/* decimal part */

	if (p_decimals > 0 || (p_decimals == -1 && (int)p_num != p_num)) {
		double dec = p_num - (double)((int)p_num);

		int digit = 0;
		if (p_decimals > MAX_DIGITS)
			p_decimals = MAX_DIGITS;

		int dec_int = 0;
		int dec_max = 0;

		while (true) {
			dec *= 10.0;
			dec_int = dec_int * 10 + (int)dec % 10;
			dec_max = dec_max * 10 + 9;
			digit++;

			if (p_decimals == -1) {
				if (digit == MAX_DIGITS) //no point in going to infinite
					break;

				if ((dec - (double)((int)dec)) < 1e-6)
					break;
			}

			if (digit == p_decimals)
				break;
		}
		dec *= 10;
		int last = (int)dec % 10;

		if (last > 5) {
			if (dec_int == dec_max) {
				dec_int = 0;
				intn++;
			} else {
				dec_int++;
			}
		}

		String decimal;
		for (int i = 0; i < digit; i++) {
			char num[2] = { 0, 0 };
			num[0] = '0' + dec_int % 10;
			decimal = num + decimal;
			dec_int /= 10;
		}
		sd = '.' + decimal;
	}

	if (intn == 0)

		s = "0";
	else {
		while (intn) {
			CharType num = '0' + (intn % 10);
			intn /= 10;
			s = num + s;
		}
	}

	s = s + sd;
	if (neg)
		s = "-" + s;
	return s;
#endif
}

String String::num_scientific(double p_num) {
	if (Math::is_nan(p_num)) {
		return "nan";
	}

	if (Math::is_inf(p_num)) {
		if (signbit(p_num)) {
			return "-inf";
		} else {
			return "inf";
		}
	}

#ifndef NO_USE_STDLIB

	char buf[256];

#if defined(__GNUC__) || defined(_MSC_VER)

#if defined(__MINGW32__) && defined(_TWO_DIGIT_EXPONENT) && !defined(_UCRT)
	// MinGW requires _set_output_format() to conform to C99 output for printf
	unsigned int old_exponent_format = _set_output_format(_TWO_DIGIT_EXPONENT);
#endif
	snprintf(buf, 256, "%lg", p_num);

#if defined(__MINGW32__) && defined(_TWO_DIGIT_EXPONENT) && !defined(_UCRT)
	_set_output_format(old_exponent_format);
#endif

#else
	sprintf(buf, "%.16lg", p_num);
#endif

	buf[255] = 0;

	return buf;
#else

	return String::num(p_num);
#endif
}

String String::num_real(double p_num) {
	if (Math::is_nan(p_num)) {
		return "nan";
	}

	if (Math::is_inf(p_num)) {
		if (signbit(p_num)) {
			return "-inf";
		} else {
			return "inf";
		}
	}

	String s;
	String sd;
	/* integer part */

	bool neg = p_num < 0;
	p_num = ABS(p_num);
	int intn = (int)p_num;

	/* decimal part */

	if ((int)p_num != p_num) {
		double dec = p_num - (double)((int)p_num);

		int digit = 0;
		int decimals = MAX_DIGITS;

		int dec_int = 0;
		int dec_max = 0;

		while (true) {
			dec *= 10.0;
			dec_int = dec_int * 10 + (int)dec % 10;
			dec_max = dec_max * 10 + 9;
			digit++;

			if ((dec - (double)((int)dec)) < 1e-6) {
				break;
			}

			if (digit == decimals) {
				break;
			}
		}

		dec *= 10;
		int last = (int)dec % 10;

		if (last > 5) {
			if (dec_int == dec_max) {
				dec_int = 0;
				intn++;
			} else {
				dec_int++;
			}
		}

		String decimal;
		for (int i = 0; i < digit; i++) {
			char num[2] = { 0, 0 };
			num[0] = '0' + dec_int % 10;
			decimal = num + decimal;
			dec_int /= 10;
		}
		sd = '.' + decimal;
	} else {
		sd = ".0";
	}

	if (intn == 0) {
		s = "0";
	} else {
		while (intn) {
			CharType num = '0' + (intn % 10);
			intn /= 10;
			s = num + s;
		}
	}

	s = s + sd;
	if (neg) {
		s = "-" + s;
	}
	return s;
}

String String::num_int64(int64_t p_num, int base, bool capitalize_hex) {
	bool sign = p_num < 0;

	int64_t n = p_num;

	int chars = 0;
	do {
		n /= base;
		chars++;
	} while (n);

	if (sign) {
		chars++;
	}
	String s;
	s.resize(chars + 1);
	CharType *c = s.ptrw();
	c[chars] = 0;
	n = p_num;
	do {
		int mod = ABS(n % base);
		if (mod >= 10) {
			char a = (capitalize_hex ? 'A' : 'a');
			c[--chars] = a + (mod - 10);
		} else {
			c[--chars] = '0' + mod;
		}

		n /= base;
	} while (n);

	if (sign) {
		c[0] = '-';
	}

	return s;
}

String String::num_uint64(uint64_t p_num, int base, bool capitalize_hex) {
	uint64_t n = p_num;

	int chars = 0;
	do {
		n /= base;
		chars++;
	} while (n);

	String s;
	s.resize(chars + 1);
	CharType *c = s.ptrw();
	c[chars] = 0;
	n = p_num;
	do {
		int mod = n % base;
		if (mod >= 10) {
			char a = (capitalize_hex ? 'A' : 'a');
			c[--chars] = a + (mod - 10);
		} else {
			c[--chars] = '0' + mod;
		}

		n /= base;
	} while (n);

	return s;
}

String String::chr(CharType p_char) {
	CharType c[2] = { p_char, 0 };
	return String(c);
}

String String::md5(const uint8_t *p_md5) {
	return String::hex_encode_buffer(p_md5, 16);
}

String String::hex_encode_buffer(const uint8_t *p_buffer, int p_len) {
	static const char hex[16] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' };

	String ret;
	char v[2] = { 0, 0 };

	for (int i = 0; i < p_len; i++) {
		v[0] = hex[p_buffer[i] >> 4];
		ret += v;
		v[0] = hex[p_buffer[i] & 0xF];
		ret += v;
	}

	return ret;
}

String String::bool_num(bool p_val) {
	if (p_val) {
		return "1";
	} else {
		return "0";
	}
}

String String::bool_str(bool p_val) {
	if (p_val) {
		return "true";
	} else {
		return "false";
	}
}

bool String::is_numeric() const {
	if (length() == 0) {
		return false;
	};

	int s = 0;
	if (operator[](0) == '-') {
		++s;
	}
	bool dot = false;
	for (int i = s; i < length(); i++) {
		CharType c = operator[](i);
		if (c == '.') {
			if (dot) {
				return false;
			};
			dot = true;
			continue;
		}
		if (c < '0' || c > '9') {
			return false;
		};
	};

	return true; // TODO: Use the parser below for this instead
};

bool String::is_zero() const {
	int size = length();

	if (size == 0) {
		return false;
	}

	int starti = 0;

	if (operator[](0) == '-') {
		starti += 1;
	}

	bool had_dot = false;
	for (int i = starti; i < size; ++i) {
		CharType c = operator[](i);

		if (c == '.') {
			if (!had_dot) {
				had_dot = true;
				continue;
			} else {
				return false;
			}
		}

		if (c != '0') {
			return false;
		}
	}

	return true;
}

String String::capitalize() const {
	String aux = this->camelcase_to_underscore(true).replace("_", " ").strip_edges();
	String cap;
	for (int i = 0; i < aux.get_slice_count(" "); i++) {
		String slice = aux.get_slicec(' ', i);
		if (slice.length() > 0) {
			slice[0] = _find_upper(slice[0]);
			if (i > 0) {
				cap += " ";
			}
			cap += slice;
		}
	}

	return cap;
}

String String::camelcase_to_underscore(bool lowercase) const {
	const CharType *cstr = get_data();
	String new_string;
	const char A = 'A', Z = 'Z';
	const char a = 'a', z = 'z';
	int start_index = 0;

	for (int i = 1; i < this->size(); i++) {
		bool is_upper = cstr[i] >= A && cstr[i] <= Z;
		bool is_number = cstr[i] >= '0' && cstr[i] <= '9';
		bool are_next_2_lower = false;
		bool is_next_lower = false;
		bool is_next_number = false;
		bool was_precedent_upper = cstr[i - 1] >= A && cstr[i - 1] <= Z;
		bool was_precedent_number = cstr[i - 1] >= '0' && cstr[i - 1] <= '9';

		if (i + 2 < this->size()) {
			are_next_2_lower = cstr[i + 1] >= a && cstr[i + 1] <= z && cstr[i + 2] >= a && cstr[i + 2] <= z;
		}

		if (i + 1 < this->size()) {
			is_next_lower = cstr[i + 1] >= a && cstr[i + 1] <= z;
			is_next_number = cstr[i + 1] >= '0' && cstr[i + 1] <= '9';
		}

		const bool cond_a = is_upper && !was_precedent_upper && !was_precedent_number;
		const bool cond_b = was_precedent_upper && is_upper && are_next_2_lower;
		const bool cond_c = is_number && !was_precedent_number;
		const bool can_break_number_letter = is_number && !was_precedent_number && is_next_lower;
		const bool can_break_letter_number = !is_number && was_precedent_number && (is_next_lower || is_next_number);

		bool should_split = cond_a || cond_b || cond_c || can_break_number_letter || can_break_letter_number;
		if (should_split) {
			new_string += this->substr(start_index, i - start_index) + "_";
			start_index = i;
		}
	}

	new_string += this->substr(start_index, this->size() - start_index);
	return lowercase ? new_string.to_lower() : new_string;
}

int String::get_slice_count(String p_splitter) const {
	if (empty()) {
		return 0;
	}
	if (p_splitter.empty()) {
		return 0;
	}

	int pos = 0;
	int slices = 1;

	while ((pos = find(p_splitter, pos)) >= 0) {
		slices++;
		pos += p_splitter.length();
	}

	return slices;
}

String String::get_slice(String p_splitter, int p_slice) const {
	if (empty() || p_splitter.empty()) {
		return "";
	}

	int pos = 0;
	int prev_pos = 0;
	//int slices=1;
	if (p_slice < 0) {
		return "";
	}
	if (find(p_splitter) == -1) {
		return *this;
	}

	int i = 0;
	while (true) {
		pos = find(p_splitter, pos);
		if (pos == -1) {
			pos = length(); //reached end
		}

		int from = prev_pos;
		//int to=pos;

		if (p_slice == i) {
			return substr(from, pos - from);
		}

		if (pos == length()) { //reached end and no find
			break;
		}
		pos += p_splitter.length();
		prev_pos = pos;
		i++;
	}

	return ""; //no find!
}

String String::get_slicec(CharType p_splitter, int p_slice) const {
	if (empty()) {
		return String();
	}

	if (p_slice < 0) {
		return String();
	}

	const CharType *c = this->ptr();
	int i = 0;
	int prev = 0;
	int count = 0;
	while (true) {
		if (c[i] == 0 || c[i] == p_splitter) {
			if (p_slice == count) {
				return substr(prev, i - prev);
			} else if (c[i] == 0) {
				return String();
			} else {
				count++;
				prev = i + 1;
			}
		}

		i++;
	}
}

Vector<String> String::split(const String &p_splitter, bool p_allow_empty, int p_maxsplit) const {
	Vector<String> ret;
	int from = 0;
	int len = length();

	while (true) {
		int end = find(p_splitter, from);
		if (end < 0) {
			end = len;
		}
		if (p_allow_empty || (end > from)) {
			if (p_maxsplit <= 0) {
				ret.push_back(substr(from, end - from));
			} else {
				// Put rest of the string and leave cycle.
				if (p_maxsplit == ret.size()) {
					ret.push_back(substr(from, len));
					break;
				}

				// Otherwise, push items until positive limit is reached.
				ret.push_back(substr(from, end - from));
			}
		}

		if (end == len) {
			break;
		}

		from = end + p_splitter.length();
	}

	return ret;
}

Vector<String> String::rsplit(const String &p_splitter, bool p_allow_empty, int p_maxsplit) const {
	Vector<String> ret;
	const int len = length();
	int remaining_len = len;

	while (true) {
		if (remaining_len < p_splitter.length() || (p_maxsplit > 0 && p_maxsplit == ret.size())) {
			// no room for another splitter or hit max splits, push what's left and we're done
			if (p_allow_empty || remaining_len > 0) {
				ret.push_back(substr(0, remaining_len));
			}
			break;
		}

		int left_edge = rfind(p_splitter, remaining_len - p_splitter.length());

		if (left_edge < 0) {
			// no more splitters, we're done
			ret.push_back(substr(0, remaining_len));
			break;
		}

		int substr_start = left_edge + p_splitter.length();
		if (p_allow_empty || substr_start < remaining_len) {
			ret.push_back(substr(substr_start, remaining_len - substr_start));
		}

		remaining_len = left_edge;
	}

	ret.invert();
	return ret;
}

Vector<String> String::split_spaces() const {
	Vector<String> ret;
	int from = 0;
	int i = 0;
	int len = length();
	if (len == 0) {
		return ret;
	}

	bool inside = false;

	while (true) {
		bool empty = operator[](i) < 33;

		if (i == 0) {
			inside = !empty;
		}

		if (!empty && !inside) {
			inside = true;
			from = i;
		}

		if (empty && inside) {
			ret.push_back(substr(from, i - from));
			inside = false;
		}

		if (i == len) {
			break;
		}
		i++;
	}

	return ret;
}

Vector<float> String::split_floats(const String &p_splitter, bool p_allow_empty) const {
	Vector<float> ret;
	int from = 0;
	int len = length();

	while (true) {
		int end = find(p_splitter, from);
		if (end < 0) {
			end = len;
		}
		if (p_allow_empty || (end > from)) {
			ret.push_back(String::to_double(&get_data()[from]));
		}

		if (end == len) {
			break;
		}

		from = end + p_splitter.length();
	}

	return ret;
}

Vector<float> String::split_floats_mk(const Vector<String> &p_splitters, bool p_allow_empty) const {
	Vector<float> ret;
	int from = 0;
	int len = length();

	while (true) {
		int idx;
		int end = findmk(p_splitters, from, &idx);
		int spl_len = 1;
		if (end < 0) {
			end = len;
		} else {
			spl_len = p_splitters[idx].length();
		}

		if (p_allow_empty || (end > from)) {
			ret.push_back(String::to_double(&get_data()[from]));
		}

		if (end == len) {
			break;
		}

		from = end + spl_len;
	}

	return ret;
}

Vector<int> String::split_ints(const String &p_splitter, bool p_allow_empty) const {
	Vector<int> ret;
	int from = 0;
	int len = length();

	while (true) {
		int end = find(p_splitter, from);
		if (end < 0) {
			end = len;
		}
		if (p_allow_empty || (end > from)) {
			ret.push_back(String::to_int(&get_data()[from], end - from));
		}

		if (end == len) {
			break;
		}

		from = end + p_splitter.length();
	}

	return ret;
}

Vector<int> String::split_ints_mk(const Vector<String> &p_splitters, bool p_allow_empty) const {
	Vector<int> ret;
	int from = 0;
	int len = length();

	while (true) {
		int idx;
		int end = findmk(p_splitters, from, &idx);
		int spl_len = 1;
		if (end < 0) {
			end = len;
		} else {
			spl_len = p_splitters[idx].length();
		}

		if (p_allow_empty || (end > from)) {
			ret.push_back(String::to_int(&get_data()[from], end - from));
		}

		if (end == len) {
			break;
		}

		from = end + spl_len;
	}

	return ret;
}

String String::join(const Vector<String> &parts) const {
	String ret;
	for (int i = 0; i < parts.size(); ++i) {
		if (i > 0) {
			ret += *this;
		}
		ret += parts[i];
	}
	return ret;
}

CharType String::char_uppercase(CharType p_char) {
	return _find_upper(p_char);
}

CharType String::char_lowercase(CharType p_char) {
	return _find_lower(p_char);
}

String String::to_lower() const {
	String lower = *this;

	for (int i = 0; i < lower.size(); i++) {
		const CharType s = lower[i];
		const CharType t = _find_lower(s);
		if (s != t) { // avoid copy on write
			lower[i] = t;
		}
	}

	return lower;
}

String String::to_upper() const {
	String upper = *this;

	for (int i = 0; i < upper.size(); i++) {
		const CharType s = upper[i];
		const CharType t = _find_upper(s);
		if (s != t) { // avoid copy on write
			upper[i] = t;
		}
	}

	return upper;
}

int String::_count(const String &p_string, int p_from, int p_to, bool p_case_insensitive) const {
	if (p_string.empty()) {
		return 0;
	}
	int len = length();
	int slen = p_string.length();
	if (len < slen) {
		return 0;
	}
	String str;
	if (p_from >= 0 && p_to >= 0) {
		if (p_to == 0) {
			p_to = len;
		} else if (p_from >= p_to) {
			return 0;
		}
		if (p_from == 0 && p_to == len) {
			str = String();
			str.copy_from_unchecked(&get_data()[0], len);
		} else {
			str = substr(p_from, p_to - p_from);
		}
	} else {
		return 0;
	}
	int c = 0;
	int idx = -1;
	do {
		idx = p_case_insensitive ? str.findn(p_string) : str.find(p_string);
		if (idx != -1) {
			str = str.substr(idx + slen, str.length() - slen);
			++c;
		}
	} while (idx != -1);
	return c;
}

int String::count(const String &p_string, int p_from, int p_to) const {
	return _count(p_string, p_from, p_to, false);
}

int String::countn(const String &p_string, int p_from, int p_to) const {
	return _count(p_string, p_from, p_to, true);
}

String String::left(int p_pos) const {
	if (p_pos <= 0) {
		return "";
	}

	if (p_pos >= length()) {
		return *this;
	}

	return substr(0, p_pos);
}

String String::right(int p_pos) const {
	if (p_pos >= length()) {
		return "";
	}

	if (p_pos <= 0) {
		return *this;
	}

	return substr(p_pos, (length() - p_pos));
}

String String::indent(const String &p_prefix) const {
	String new_string;
	int line_start = 0;

	for (int i = 0; i < length(); i++) {
		const CharType c = operator[](i);
		if (c == '\n') {
			if (i == line_start) {
				new_string += c; // Leave empty lines empty.
			} else {
				new_string += p_prefix + substr(line_start, i - line_start + 1);
			}
			line_start = i + 1;
		}
	}
	if (line_start != length()) {
		new_string += p_prefix + substr(line_start);
	}
	return new_string;
}

String String::dedent() const {
	String new_string;
	String indent;
	bool has_indent = false;
	bool has_text = false;
	int line_start = 0;
	int indent_stop = -1;

	for (int i = 0; i < length(); i++) {
		CharType c = operator[](i);
		if (c == '\n') {
			if (has_text) {
				new_string += substr(indent_stop, i - indent_stop);
			}
			new_string += "\n";
			has_text = false;
			line_start = i + 1;
			indent_stop = -1;
		} else if (!has_text) {
			if (c > 32) {
				has_text = true;
				if (!has_indent) {
					has_indent = true;
					indent = substr(line_start, i - line_start);
					indent_stop = i;
				}
			}
			if (has_indent && indent_stop < 0) {
				int j = i - line_start;
				if (j >= indent.length() || c != indent[j]) {
					indent_stop = i;
				}
			}
		}
	}

	if (has_text) {
		new_string += substr(indent_stop, length() - indent_stop);
	}

	return new_string;
}

String String::strip_edges(bool left, bool right) const {
	int len = length();
	int beg = 0, end = len;

	if (left) {
		for (int i = 0; i < len; i++) {
			if (operator[](i) <= 32) {
				beg++;
			} else {
				break;
			}
		}
	}

	if (right) {
		for (int i = (int)(len - 1); i >= 0; i--) {
			if (operator[](i) <= 32) {
				end--;
			} else {
				break;
			}
		}
	}

	if (beg == 0 && end == len) {
		return *this;
	}

	return substr(beg, end - beg);
}

String String::strip_escapes() const {
	String new_string;
	for (int i = 0; i < length(); i++) {
		// Escape characters on first page of the ASCII table, before 32 (Space).
		if (operator[](i) < 32) {
			continue;
		}
		new_string += operator[](i);
	}

	return new_string;
}

String String::lstrip(const String &p_chars) const {
	int len = length();
	int beg;

	for (beg = 0; beg < len; beg++) {
		if (p_chars.find_char(get(beg)) == -1) {
			break;
		}
	}

	if (beg == 0) {
		return *this;
	}

	return substr(beg, len - beg);
}

String String::rstrip(const String &p_chars) const {
	int len = length();
	int end;

	for (end = len - 1; end >= 0; end--) {
		if (p_chars.find_char(get(end)) == -1) {
			break;
		}
	}

	if (end == len - 1) {
		return *this;
	}

	return substr(0, end + 1);
}

String String::get_extension() const {
	int pos = rfind(".");
	if (pos < 0 || pos < MAX(rfind("/"), rfind("\\"))) {
		return "";
	}

	return substr(pos + 1, length());
}

String String::get_basename() const {
	int pos = rfind(".");
	if (pos < 0 || pos < MAX(rfind("/"), rfind("\\"))) {
		return *this;
	}

	return substr(0, pos);
}

String String::plus_file(const String &p_file) const {
	if (empty()) {
		return p_file;
	}
	if (operator[](length() - 1) == '/' || (p_file.size() > 0 && p_file.operator[](0) == '/')) {
		return *this + p_file;
	}
	return *this + "/" + p_file;
}

CharType String::unicode_at(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, length(), 0);
	return operator[](p_idx);
}

CharType String::ord_at(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, length(), 0);
	return operator[](p_idx);
}

void String::erase(int p_pos, int p_chars) {
	*this = left(p_pos) + substr(p_pos + p_chars, length() - ((p_pos + p_chars)));
}
CharString String::ascii(bool p_allow_extended) const {
	if (!length()) {
		return CharString();
	}

	CharString cs;
	cs.resize(size());

	for (int i = 0; i < size(); i++) {
		CharType c = operator[](i);
		if ((c <= 0x7f) || (c <= 0xff && p_allow_extended)) {
			cs[i] = c;
		} else {
			//print_unicode_error(vformat("Invalid unicode codepoint (%x), cannot represent as ASCII/Latin-1", (uint32_t)c));
			cs[i] = 0x20;
		}
	}

	return cs;
}

String String::utf8(const char *p_utf8, int p_len) {
	String ret;
	ret.parse_utf8(p_utf8, p_len);

	return ret;
}

Error String::parse_utf8(const char *p_utf8, int p_len, bool p_skip_cr) {
	if (!p_utf8) {
		return ERR_INVALID_DATA;
	}

	String aux;

	int cstr_size = 0;
	int str_size = 0;

	/* HANDLE BOM (Byte Order Mark) */
	if (p_len < 0 || p_len >= 3) {
		bool has_bom = uint8_t(p_utf8[0]) == 0xef && uint8_t(p_utf8[1]) == 0xbb && uint8_t(p_utf8[2]) == 0xbf;
		if (has_bom) {
			//8-bit encoding, byte order has no meaning in UTF-8, just skip it
			if (p_len >= 0) {
				p_len -= 3;
			}
			p_utf8 += 3;
		}
	}

	bool decode_error = false;
	bool decode_failed = false;
	{
		const char *ptrtmp = p_utf8;
		const char *ptrtmp_limit = &p_utf8[p_len];
		int skip = 0;
		uint8_t c_start = 0;
		while (ptrtmp != ptrtmp_limit && *ptrtmp) {
			uint8_t c = *ptrtmp >= 0 ? *ptrtmp : uint8_t(256 + *ptrtmp);

			if (skip == 0) {
				if (p_skip_cr && c == '\r') {
					ptrtmp++;
					continue;
				}
				/* Determine the number of characters in sequence */
				if ((c & 0x80) == 0) {
					skip = 0;
				} else if ((c & 0xe0) == 0xc0) {
					skip = 1;
				} else if ((c & 0xf0) == 0xe0) {
					skip = 2;
				} else if ((c & 0xf8) == 0xf0) {
					skip = 3;
				} else if ((c & 0xfc) == 0xf8) {
					skip = 4;
				} else if ((c & 0xfe) == 0xfc) {
					skip = 5;
				} else {
					skip = 0;
					//print_unicode_error(vformat("Invalid UTF-8 leading byte (%x)", c), true);
					decode_failed = true;
				}
				c_start = c;

				if (skip == 1 && (c & 0x1e) == 0) {
					//print_unicode_error(vformat("Overlong encoding (%x ...)", c));
					decode_error = true;
				}
				str_size++;
			} else {
				if ((c_start == 0xe0 && skip == 2 && c < 0xa0) || (c_start == 0xf0 && skip == 3 && c < 0x90) || (c_start == 0xf8 && skip == 4 && c < 0x88) || (c_start == 0xfc && skip == 5 && c < 0x84)) {
					//print_unicode_error(vformat("Overlong encoding (%x %x ...)", c_start, c));
					decode_error = true;
				}
				if (c < 0x80 || c > 0xbf) {
					//print_unicode_error(vformat("Invalid UTF-8 continuation byte (%x ... %x ...)", c_start, c), true);
					decode_failed = true;
					skip = 0;
				} else {
					--skip;
				}
			}

			cstr_size++;
			ptrtmp++;
		}

		if (skip) {
			//rrprint_unicode_error(vformat("Missing %d UTF-8 continuation byte(s)", skip), true);
			decode_failed = true;
		}
	}

	if (str_size == 0) {
		clear();
		return OK; // empty string
	}

	resize(str_size + 1);
	CharType *dst = ptrw();
	dst[str_size] = 0;

	int skip = 0;
	uint32_t unichar = 0;
	while (cstr_size) {
		uint8_t c = *p_utf8 >= 0 ? *p_utf8 : uint8_t(256 + *p_utf8);

		if (skip == 0) {
			if (p_skip_cr && c == '\r') {
				p_utf8++;
				continue;
			}
			/* Determine the number of characters in sequence */
			if ((c & 0x80) == 0) {
				*(dst++) = c;
				unichar = 0;
				skip = 0;
			} else if ((c & 0xe0) == 0xc0) {
				unichar = (0xff >> 3) & c;
				skip = 1;
			} else if ((c & 0xf0) == 0xe0) {
				unichar = (0xff >> 4) & c;
				skip = 2;
			} else if ((c & 0xf8) == 0xf0) {
				unichar = (0xff >> 5) & c;
				skip = 3;
			} else if ((c & 0xfc) == 0xf8) {
				unichar = (0xff >> 6) & c;
				skip = 4;
			} else if ((c & 0xfe) == 0xfc) {
				unichar = (0xff >> 7) & c;
				skip = 5;
			} else {
				*(dst++) = 0x20;
				unichar = 0;
				skip = 0;
			}
		} else {
			if (c < 0x80 || c > 0xbf) {
				*(dst++) = 0x20;
				skip = 0;
			} else {
				unichar = (unichar << 6) | (c & 0x3f);
				--skip;
				if (skip == 0) {
					if (unichar == 0) {
						//print_unicode_error("NUL character", true);
						decode_failed = true;
						unichar = 0x20;
					}
					if ((unichar & 0xfffff800) == 0xd800) {
						//print_unicode_error(vformat("Unpaired surrogate (%x)", unichar));
						decode_error = true;
					}
					if (unichar > 0x10ffff) {
						//print_unicode_error(vformat("Invalid unicode codepoint (%x)", unichar));
						decode_error = true;
					}
					*(dst++) = unichar;
				}
			}
		}

		cstr_size--;
		p_utf8++;
	}
	if (skip) {
		*(dst++) = 0x20;
	}

	if (decode_failed) {
		return ERR_INVALID_DATA;
	} else if (decode_error) {
		return ERR_PARSE_ERROR;
	} else {
		return OK;
	}
}

CharString String::utf8() const {
	int l = length();
	if (!l) {
		return CharString();
	}

	const CharType *d = &operator[](0);
	int fl = 0;
	for (int i = 0; i < l; i++) {
		uint32_t c = d[i];
		if (c <= 0x7f) { // 7 bits.
			fl += 1;
		} else if (c <= 0x7ff) { // 11 bits
			fl += 2;
		} else if (c <= 0xffff) { // 16 bits
			fl += 3;
		} else if (c <= 0x001fffff) { // 21 bits
			fl += 4;
		} else if (c <= 0x03ffffff) { // 26 bits
			fl += 5;
			//print_unicode_error(vformat("Invalid unicode codepoint (%x)", c));
		} else if (c <= 0x7fffffff) { // 31 bits
			fl += 6;
			//print_unicode_error(vformat("Invalid unicode codepoint (%x)", c));
		} else {
			fl += 1;
			//print_unicode_error(vformat("Invalid unicode codepoint (%x), cannot represent as UTF-8", c), true);
		}
	}

	CharString utf8s;
	if (fl == 0) {
		return utf8s;
	}

	utf8s.resize(fl + 1);
	uint8_t *cdst = (uint8_t *)utf8s.get_data();

#define APPEND_CHAR(m_c) *(cdst++) = m_c

	for (int i = 0; i < l; i++) {
		uint32_t c = d[i];

		if (c <= 0x7f) { // 7 bits.
			APPEND_CHAR(c);
		} else if (c <= 0x7ff) { // 11 bits
			APPEND_CHAR(uint32_t(0xc0 | ((c >> 6) & 0x1f))); // Top 5 bits.
			APPEND_CHAR(uint32_t(0x80 | (c & 0x3f))); // Bottom 6 bits.
		} else if (c <= 0xffff) { // 16 bits
			APPEND_CHAR(uint32_t(0xe0 | ((c >> 12) & 0x0f))); // Top 4 bits.
			APPEND_CHAR(uint32_t(0x80 | ((c >> 6) & 0x3f))); // Middle 6 bits.
			APPEND_CHAR(uint32_t(0x80 | (c & 0x3f))); // Bottom 6 bits.
		} else if (c <= 0x001fffff) { // 21 bits
			APPEND_CHAR(uint32_t(0xf0 | ((c >> 18) & 0x07))); // Top 3 bits.
			APPEND_CHAR(uint32_t(0x80 | ((c >> 12) & 0x3f))); // Upper middle 6 bits.
			APPEND_CHAR(uint32_t(0x80 | ((c >> 6) & 0x3f))); // Lower middle 6 bits.
			APPEND_CHAR(uint32_t(0x80 | (c & 0x3f))); // Bottom 6 bits.
		} else if (c <= 0x03ffffff) { // 26 bits
			APPEND_CHAR(uint32_t(0xf8 | ((c >> 24) & 0x03))); // Top 2 bits.
			APPEND_CHAR(uint32_t(0x80 | ((c >> 18) & 0x3f))); // Upper middle 6 bits.
			APPEND_CHAR(uint32_t(0x80 | ((c >> 12) & 0x3f))); // middle 6 bits.
			APPEND_CHAR(uint32_t(0x80 | ((c >> 6) & 0x3f))); // Lower middle 6 bits.
			APPEND_CHAR(uint32_t(0x80 | (c & 0x3f))); // Bottom 6 bits.
		} else if (c <= 0x7fffffff) { // 31 bits
			APPEND_CHAR(uint32_t(0xfc | ((c >> 30) & 0x01))); // Top 1 bit.
			APPEND_CHAR(uint32_t(0x80 | ((c >> 24) & 0x3f))); // Upper upper middle 6 bits.
			APPEND_CHAR(uint32_t(0x80 | ((c >> 18) & 0x3f))); // Lower upper middle 6 bits.
			APPEND_CHAR(uint32_t(0x80 | ((c >> 12) & 0x3f))); // Upper lower middle 6 bits.
			APPEND_CHAR(uint32_t(0x80 | ((c >> 6) & 0x3f))); // Lower lower middle 6 bits.
			APPEND_CHAR(uint32_t(0x80 | (c & 0x3f))); // Bottom 6 bits.
		} else {
			APPEND_CHAR(0x20);
		}
	}
#undef APPEND_CHAR
	*cdst = 0; //trailing zero

	return utf8s;
}

int String::utf8_byte_length() const {
	int l = length();
	if (!l) {
		return 0;
	}

	const CharType *d = &operator[](0);
	int fl = 0;
	for (int i = 0; i < l; i++) {
		uint32_t c = d[i];
		if (c <= 0x7f) { // 7 bits.
			fl += 1;
		} else if (c <= 0x7ff) { // 11 bits
			fl += 2;
		} else if (c <= 0xffff) { // 16 bits
			fl += 3;
		} else if (c <= 0x001fffff) { // 21 bits
			fl += 4;
		} else if (c <= 0x03ffffff) { // 26 bits
			fl += 5;
			//print_unicode_error(vformat("Invalid unicode codepoint (%x)", c));
		} else if (c <= 0x7fffffff) { // 31 bits
			fl += 6;
			//print_unicode_error(vformat("Invalid unicode codepoint (%x)", c));
		} else {
			fl += 1;
			//print_unicode_error(vformat("Invalid unicode codepoint (%x), cannot represent as UTF-8", c), true);
		}
	}

	return fl;
}

String String::utf16(const char16_t *p_utf16, int p_len) {
	String ret;
	ret.parse_utf16(p_utf16, p_len);

	return ret;
}

Error String::parse_utf16(const char16_t *p_utf16, int p_len) {
	if (!p_utf16) {
		return ERR_INVALID_DATA;
	}

	String aux;

	int cstr_size = 0;
	int str_size = 0;

	/* HANDLE BOM (Byte Order Mark) */
	bool byteswap = false; // assume correct endianness if no BOM found
	if (p_len < 0 || p_len >= 1) {
		bool has_bom = false;
		if (uint16_t(p_utf16[0]) == 0xfeff) { // correct BOM, read as is
			has_bom = true;
			byteswap = false;
		} else if (uint16_t(p_utf16[0]) == 0xfffe) { // backwards BOM, swap bytes
			has_bom = true;
			byteswap = true;
		}
		if (has_bom) {
			if (p_len >= 0) {
				p_len -= 1;
			}
			p_utf16 += 1;
		}
	}

	bool decode_error = false;
	{
		const char16_t *ptrtmp = p_utf16;
		const char16_t *ptrtmp_limit = &p_utf16[p_len];
		//uint32_t c_prev = 0;
		bool skip = false;
		while (ptrtmp != ptrtmp_limit && *ptrtmp) {
			uint32_t c = (byteswap) ? BSWAP16(*ptrtmp) : *ptrtmp;

			if ((c & 0xfffffc00) == 0xd800) { // lead surrogate
				if (skip) {
					//print_unicode_error(vformat("Unpaired lead surrogate (%x [trail?] %x)", c_prev, c));
					decode_error = true;
				}
				skip = true;
			} else if ((c & 0xfffffc00) == 0xdc00) { // trail surrogate
				if (skip) {
					str_size--;
				} else {
					//print_unicode_error(vformat("Unpaired trail surrogate (%x [lead?] %x)", c_prev, c));
					decode_error = true;
				}
				skip = false;
			} else {
				skip = false;
			}

			//c_prev = c;
			str_size++;
			cstr_size++;
			ptrtmp++;
		}

		if (skip) {
			//print_unicode_error(vformat("Unpaired lead surrogate (%x [eol])", c_prev));
			decode_error = true;
		}
	}

	if (str_size == 0) {
		clear();
		return OK; // empty string
	}

	resize(str_size + 1);
	CharType *dst = ptrw();
	dst[str_size] = 0;

	bool skip = false;
	uint32_t c_prev = 0;
	while (cstr_size) {
		uint32_t c = (byteswap) ? BSWAP16(*p_utf16) : *p_utf16;

		if ((c & 0xfffffc00) == 0xd800) { // lead surrogate
			if (skip) {
				*(dst++) = c_prev; // unpaired, store as is
			}
			skip = true;
		} else if ((c & 0xfffffc00) == 0xdc00) { // trail surrogate
			if (skip) {
				*(dst++) = (c_prev << 10UL) + c - ((0xd800 << 10UL) + 0xdc00 - 0x10000); // decode pair
			} else {
				*(dst++) = c; // unpaired, store as is
			}
			skip = false;
		} else {
			*(dst++) = c;
			skip = false;
		}

		cstr_size--;
		p_utf16++;
		c_prev = c;
	}

	if (skip) {
		*(dst++) = c_prev;
	}

	if (decode_error) {
		return ERR_PARSE_ERROR;
	} else {
		return OK;
	}
}

Char16String String::utf16() const {
	int l = length();
	if (!l) {
		return Char16String();
	}

	const CharType *d = &operator[](0);
	int fl = 0;
	for (int i = 0; i < l; i++) {
		uint32_t c = d[i];
		if (c <= 0xffff) { // 16 bits.
			fl += 1;
			if ((c & 0xfffff800) == 0xd800) {
				//print_unicode_error(vformat("Unpaired surrogate (%x)", c));
			}
		} else if (c <= 0x10ffff) { // 32 bits.
			fl += 2;
		} else {
			//print_unicode_error(vformat("Invalid unicode codepoint (%x), cannot represent as UTF-16", c), true);
			fl += 1;
		}
	}

	Char16String utf16s;
	if (fl == 0) {
		return utf16s;
	}

	utf16s.resize(fl + 1);
	uint16_t *cdst = (uint16_t *)utf16s.get_data();

#define APPEND_CHAR(m_c) *(cdst++) = m_c

	for (int i = 0; i < l; i++) {
		uint32_t c = d[i];

		if (c <= 0xffff) { // 16 bits.
			APPEND_CHAR(c);
		} else if (c <= 0x10ffff) { // 32 bits.
			APPEND_CHAR(uint32_t((c >> 10) + 0xd7c0)); // lead surrogate.
			APPEND_CHAR(uint32_t((c & 0x3ff) | 0xdc00)); // trail surrogate.
		} else {
			APPEND_CHAR(0x20);
		}
	}
#undef APPEND_CHAR
	*cdst = 0; //trailing zero

	return utf16s;
}

int String::utf16_byte_length() const {
	int l = length();
	if (!l) {
		return 0;
	}

	const CharType *d = &operator[](0);
	int fl = 0;
	for (int i = 0; i < l; i++) {
		uint32_t c = d[i];
		if (c <= 0xffff) { // 16 bits.
			fl += 1;
			if ((c & 0xfffff800) == 0xd800) {
				//print_unicode_error(vformat("Unpaired surrogate (%x)", c));
			}
		} else if (c <= 0x10ffff) { // 32 bits.
			fl += 2;
		} else {
			//print_unicode_error(vformat("Invalid unicode codepoint (%x), cannot represent as UTF-16", c), true);
			fl += 1;
		}
	}

	return fl;
}

uint32_t String::hash(const char *p_cstr) {
	uint32_t hashv = 5381;
	uint32_t c;

	while ((c = *p_cstr++)) {
		hashv = ((hashv << 5) + hashv) + c; /* hash * 33 + c */
	}

	return hashv;
}

uint32_t String::hash(const char *p_cstr, int p_len) {
	uint32_t hashv = 5381;
	for (int i = 0; i < p_len; i++) {
		hashv = ((hashv << 5) + hashv) + p_cstr[i]; /* hash * 33 + c */
	}

	return hashv;
}

uint32_t String::hash(const wchar_t *p_cstr) {
	uint32_t hashv = 5381;
	uint32_t c;

	while ((c = *p_cstr++)) {
		hashv = ((hashv << 5) + hashv) + c; /* hash * 33 + c */
	}

	return hashv;
}

uint32_t String::hash(const wchar_t *p_cstr, int p_len) {
	uint32_t hashv = 5381;
	for (int i = 0; i < p_len; i++) {
		hashv = ((hashv << 5) + hashv) + p_cstr[i]; /* hash * 33 + c */
	}

	return hashv;
}

uint32_t String::hash(const CharType *p_cstr) {
	uint32_t hashv = 5381;
	uint32_t c;

	while ((c = *p_cstr++)) {
		hashv = ((hashv << 5) + hashv) + c; /* hash * 33 + c */
	}

	return hashv;
}

uint32_t String::hash(const CharType *p_cstr, int p_len) {
	uint32_t hashv = 5381;
	for (int i = 0; i < p_len; i++) {
		hashv = ((hashv << 5) + hashv) + p_cstr[i]; /* hash * 33 + c */
	}

	return hashv;
}

uint32_t String::hash() const {
	/* simple djb2 hashing */

	const CharType *chr = get_data();
	uint32_t hashv = 5381;
	uint32_t c;

	while ((c = *chr++)) {
		hashv = ((hashv << 5) + hashv) + c; /* hash * 33 + c */
	}

	return hashv;
}

uint64_t String::hash64() const {
	/* simple djb2 hashing */

	const CharType *chr = get_data();
	uint64_t hashv = 5381;
	uint64_t c;

	while ((c = *chr++)) {
		hashv = ((hashv << 5) + hashv) + c; /* hash * 33 + c */
	}

	return hashv;
}

bool String::is_abs_path() const {
	if (length() > 1) {
		return (operator[](0) == '/' || operator[](0) == '\\' || find(":/") != -1 || find(":\\") != -1);
	} else if ((length()) == 1) {
		return (operator[](0) == '/' || operator[](0) == '\\');
	} else {
		return false;
	}
}

bool String::is_rel_path() const {
	return !is_abs_path();
}

bool String::is_resource_file() const {
	return begins_with("res://") && find("::") == -1;
}

String String::path_to(const String &p_path) const {
	String src = this->replace("\\", "/");
	String dst = p_path.replace("\\", "/");
	if (!src.ends_with("/")) {
		src += "/";
	}
	if (!dst.ends_with("/")) {
		dst += "/";
	}

	String base;

	if (src.begins_with("res://") && dst.begins_with("res://")) {
		base = "res:/";
		src = src.replace("res://", "/");
		dst = dst.replace("res://", "/");

	} else if (src.begins_with("user://") && dst.begins_with("user://")) {
		base = "user:/";
		src = src.replace("user://", "/");
		dst = dst.replace("user://", "/");

	} else if (src.begins_with("/") && dst.begins_with("/")) {
		//nothing
	} else {
		//dos style
		String src_begin = src.get_slicec('/', 0);
		String dst_begin = dst.get_slicec('/', 0);

		if (src_begin != dst_begin) {
			return p_path; //impossible to do this
		}

		base = src_begin;
		src = src.substr(src_begin.length(), src.length());
		dst = dst.substr(dst_begin.length(), dst.length());
	}

	//remove leading and trailing slash and split
	Vector<String> src_dirs = src.substr(1, src.length() - 2).split("/");
	Vector<String> dst_dirs = dst.substr(1, dst.length() - 2).split("/");

	//find common parent
	int common_parent = 0;

	while (true) {
		if (src_dirs.size() == common_parent) {
			break;
		}
		if (dst_dirs.size() == common_parent) {
			break;
		}
		if (src_dirs[common_parent] != dst_dirs[common_parent]) {
			break;
		}
		common_parent++;
	}

	common_parent--;

	String dir;

	for (int i = src_dirs.size() - 1; i > common_parent; i--) {
		dir += "../";
	}

	for (int i = common_parent + 1; i < dst_dirs.size(); i++) {
		dir += dst_dirs[i] + "/";
	}

	if (dir.length() == 0) {
		dir = "./";
	}
	return dir;
}

String String::path_to_file(const String &p_path) const {
	// Don't get base dir for src, this is expected to be a dir already.
	String src = this->replace("\\", "/");
	String dst = p_path.replace("\\", "/").get_base_dir();
	String rel = src.path_to(dst);
	if (rel == dst) { // failed
		return p_path;
	} else {
		return rel + p_path.get_file();
	}
}

String String::get_base_dir() const {
	int end = 0;

	// url scheme style base
	int basepos = find("://");
	if (basepos != -1) {
		end = basepos + 3;
	}

	// windows top level directory base
	if (end == 0) {
		basepos = find(":/");
		if (basepos == -1) {
			basepos = find(":\\");
		}
		if (basepos != -1) {
			end = basepos + 2;
		}
	}

	// Windows UNC network share path.
	if (end == 0) {
		if (is_network_share_path()) {
			basepos = find("/", 2);
			if (basepos == -1) {
				basepos = find("\\", 2);
			}
			int servpos = find("/", basepos + 1);
			if (servpos == -1) {
				servpos = find("\\", basepos + 1);
			}
			if (servpos != -1) {
				end = servpos + 1;
			}
		}
	}

	// unix root directory base
	if (end == 0) {
		if (begins_with("/")) {
			end = 1;
		}
	}

	String rs;
	String base;
	if (end != 0) {
		rs = substr(end, length());
		base = substr(0, end);
	} else {
		rs = *this;
	}

	int sep = MAX(rs.rfind("/"), rs.rfind("\\"));
	if (sep == -1) {
		return base;
	}

	return base + rs.substr(0, sep);
}

String String::get_file() const {
	int sep = MAX(rfind("/"), rfind("\\"));
	if (sep == -1) {
		return *this;
	}

	return substr(sep + 1, length());
}

String String::humanize_size(uint64_t p_size) {
	uint64_t _div = 1;
	Vector<String> prefixes;
	prefixes.push_back(RTR("B"));
	prefixes.push_back(RTR("KiB"));
	prefixes.push_back(RTR("MiB"));
	prefixes.push_back(RTR("GiB"));
	prefixes.push_back(RTR("TiB"));
	prefixes.push_back(RTR("PiB"));
	prefixes.push_back(RTR("EiB"));

	int prefix_idx = 0;

	while (prefix_idx < prefixes.size() - 1 && p_size > (_div * 1024)) {
		_div *= 1024;
		prefix_idx++;
	}

	const int digits = prefix_idx > 0 ? _humanize_digits(p_size / _div) : 0;
	const double divisor = prefix_idx > 0 ? _div : 1;

	return String::num(p_size / divisor).pad_decimals(digits) + " " + prefixes[prefix_idx];
}

String String::simplify_path() const {
	String s = *this;
	String drive;
	if (s.begins_with("local://")) {
		drive = "local://";
		s = s.substr(8, s.length());
	} else if (s.begins_with("res://")) {
		drive = "res://";
		s = s.substr(6, s.length());
	} else if (s.begins_with("user://")) {
		drive = "user://";
		s = s.substr(7, s.length());
	} else if (is_network_share_path()) {
		drive = s.substr(0, 2);
		s = s.substr(2, s.length() - 2);
	} else if (s.begins_with("/") || s.begins_with("\\")) {
		drive = s.substr(0, 1);
		s = s.substr(1, s.length() - 1);
	} else {
		int p = s.find(":/");
		if (p == -1) {
			p = s.find(":\\");
		}
		if (p != -1 && p < s.find("/")) {
			drive = s.substr(0, p + 2);
			s = s.substr(p + 2, s.length());
		}
	}

	s = s.replace("\\", "/");
	while (true) { // in case of using 2 or more slash
		String compare = s.replace("//", "/");
		if (s == compare) {
			break;
		} else {
			s = compare;
		}
	}
	Vector<String> dirs = s.split("/", false);

	for (int i = 0; i < dirs.size(); i++) {
		String d = dirs[i];
		if (d == ".") {
			dirs.remove(i);
			i--;
		} else if (d == "..") {
			if (i == 0) {
				dirs.remove(i);
				i--;
			} else {
				dirs.remove(i);
				dirs.remove(i - 1);
				i -= 2;
			}
		}
	}

	s = "";

	for (int i = 0; i < dirs.size(); i++) {
		if (i > 0) {
			s += "/";
		}
		s += dirs[i];
	}

	return drive + s;
}

bool String::is_network_share_path() const {
	return begins_with("//") || begins_with("\\\\");
}

String String::append_path(const char *path) const {
	if (path[0] == '\0') {
		return *this;
	}

	String ret = *this;
	int size = length();

	if (size == 0) {
		ret += path;
		return ret;
	}

	int sindex = 0;
	char ch = path[sindex];
	while (ch == '/' || ch == '\\') {
		if (ch == '\0') {
			return ret;
		}

		ch = path[++sindex];
	}

	// /////folder
	//      ^ (sindex)

	if (ret.ends_with("/") || ret.ends_with("\\")) {
		ret += &path[sindex];
	} else {
		if (sindex > 0) {
			ret += '/';
			ret += &path[sindex - 1];
		} else {
			ret += '/';
			ret += &path[sindex];
		}
	}

	return ret;
}

String String::append_path(const String &path) const {
	if (path.length() == 0) {
		return *this;
	}

	int size = length();

	if (size == 0) {
		return path;
	}

	int sindex = 0;
	int ts = path.size() - 1;
	char ch = path[sindex];
	while (ch == '/' || ch == '\\') {
		if (sindex == ts) {
			return *this;
		}

		ch = path[++sindex];
	}

	String ret = *this;

	// /////folder
	//      ^ (sindex)

	if (ret.ends_with("/") || ret.ends_with("\\")) {
		ret += &path[sindex];
	} else {
		if (sindex > 0) {
			ret += '/';
			ret += &path[sindex - 1];
		} else {
			ret += '/';
			ret += &path[sindex];
		}
	}

	return ret;
}

String String::path_clean_end_slash() const {
	// _size > 1, so if root is given ("/"), it will not be removed

	String ret = *this;

	int strip_to = ret.length() - 1;

	CharType c = ret[strip_to];

	while (strip_to > 1 && (c == '/' || c == '\\')) {
		--strip_to;
		c = ret[strip_to];
	}

	if (ret.length() != strip_to + 1) {
		ret.set_length(strip_to + 1);
	}

	return ret;
}
String String::path_ensure_end_slash() const {
	// Don't add if empty string, as it would make it root on linux, which can easily become a serious bug

	String ret = *this;

	if (ret.length() == 0) {
		return ret;
	}

	if (!(ret.ends_with("/") || ret.ends_with("\\"))) {
		ret += "/";
	}

	return ret;
}

String String::path_get_prev_dir() const {
	int size = length();

	if (size == 0) {
		return "/";
	}

	int seind = size - 1;
	while (seind > 0 && (operator[](seind) == '/' || operator[](seind) == '\\')) {
		--seind;
	}

	if (seind == 0) {
		// /////////
		// or
		// a///////
		// no prev dir

		return "/";
	}

	// fol/fol2/fol3//
	//             ^  (seind)

	while (seind > 0 && (operator[](seind) != '/' && operator[](seind) != '\\')) {
		--seind;
	}

	// fol/fol2/fol3//
	//         ^  (seind)

	//--seind;

	if (seind <= 0) {
		return "/";
	}

	return substr_index(0, seind);
}

String String::xml_escape(bool p_escape_quotes) const {
	String str = *this;
	str = str.replace("&", "&amp;");
	str = str.replace("<", "&lt;");
	str = str.replace(">", "&gt;");
	if (p_escape_quotes) {
		str = str.replace("'", "&apos;");
		str = str.replace("\"", "&quot;");
	}
	/*
	for (int i=1;i<32;i++) {

		char chr[2]={i,0};
		str=str.replace(chr,"&#"+String::num(i)+";");
	}*/
	return str;
}

static _FORCE_INLINE_ int _xml_unescape(const CharType *p_src, int p_src_len, CharType *p_dst) {
	int len = 0;
	while (p_src_len) {
		if (*p_src == '&') {
			int eat = 0;

			if (p_src_len >= 4 && p_src[1] == '#') {
				CharType c = 0;
				bool overflow = false;
				if (p_src[2] == 'x') {
					// Hex entity &#x<num>;
					for (int i = 3; i < p_src_len; i++) {
						eat = i + 1;
						CharType ct = p_src[i];
						if (ct == ';') {
							break;
						} else if (ct >= '0' && ct <= '9') {
							ct = ct - '0';
						} else if (ct >= 'a' && ct <= 'f') {
							ct = (ct - 'a') + 10;
						} else if (ct >= 'A' && ct <= 'F') {
							ct = (ct - 'A') + 10;
						} else {
							break;
						}
						if (c > (WCHAR_MAX >> 4)) {
							overflow = true;
							break;
						}
						c <<= 4;
						c |= ct;
					}
				} else {
					// Decimal entity &#<num>;
					for (int i = 2; i < p_src_len; i++) {
						eat = i + 1;
						CharType ct = p_src[i];
						if (ct == ';' || ct < '0' || ct > '9') {
							break;
						}
					}
					if (p_src[eat - 1] == ';') {
						int64_t val = String::to_int(p_src + 2, eat - 3);
						if (val > 0 && val <= WCHAR_MAX) {
							c = (CharType)val;
						} else {
							overflow = true;
						}
					}
				}

				// Value must be non-zero, in the range of CharType,
				// actually end with ';'. If invalid, leave the entity as-is
				if (c == '\0' || overflow || p_src[eat - 1] != ';') {
					eat = 1;
					c = *p_src;
				}
				if (p_dst) {
					*p_dst = c;
				}
			} else if (p_src_len >= 4 && p_src[1] == 'g' && p_src[2] == 't' && p_src[3] == ';') {
				if (p_dst) {
					*p_dst = '>';
				}
				eat = 4;
			} else if (p_src_len >= 4 && p_src[1] == 'l' && p_src[2] == 't' && p_src[3] == ';') {
				if (p_dst) {
					*p_dst = '<';
				}
				eat = 4;
			} else if (p_src_len >= 5 && p_src[1] == 'a' && p_src[2] == 'm' && p_src[3] == 'p' && p_src[4] == ';') {
				if (p_dst) {
					*p_dst = '&';
				}
				eat = 5;
			} else if (p_src_len >= 6 && p_src[1] == 'q' && p_src[2] == 'u' && p_src[3] == 'o' && p_src[4] == 't' && p_src[5] == ';') {
				if (p_dst) {
					*p_dst = '"';
				}
				eat = 6;
			} else if (p_src_len >= 6 && p_src[1] == 'a' && p_src[2] == 'p' && p_src[3] == 'o' && p_src[4] == 's' && p_src[5] == ';') {
				if (p_dst) {
					*p_dst = '\'';
				}
				eat = 6;
			} else {
				if (p_dst) {
					*p_dst = *p_src;
				}
				eat = 1;
			}

			if (p_dst) {
				p_dst++;
			}

			len++;
			p_src += eat;
			p_src_len -= eat;
		} else {
			if (p_dst) {
				*p_dst = *p_src;
				p_dst++;
			}
			len++;
			p_src++;
			p_src_len--;
		}
	}

	return len;
}

String String::xml_unescape() const {
	String str;
	int l = length();
	int len = _xml_unescape(get_data(), l, nullptr);
	if (len == 0) {
		return String();
	}
	str.resize(len + 1);
	_xml_unescape(get_data(), l, str.ptrw());
	str[len] = 0;
	return str;
}

String String::http_escape() const {
	const CharString temp = utf8();
	String res;
	for (int i = 0; i < temp.length(); ++i) {
		uint8_t ord = temp[i];
		if (ord == '.' || ord == '-' || ord == '_' || ord == '~' ||
				(ord >= 'a' && ord <= 'z') ||
				(ord >= 'A' && ord <= 'Z') ||
				(ord >= '0' && ord <= '9')) {
			res += ord;
		} else {
			char p[4] = { '%', 0, 0, 0 };
			static const char hex[16] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F' };

			p[1] = hex[ord >> 4];
			p[2] = hex[ord & 0xF];
			res += p;
		}
	}
	return res;
}

String String::http_unescape() const {
	String res;
	for (int i = 0; i < length(); ++i) {
		if (ord_at(i) == '%' && i + 2 < length()) {
			CharType ord1 = ord_at(i + 1);
			if ((ord1 >= '0' && ord1 <= '9') || (ord1 >= 'A' && ord1 <= 'Z')) {
				CharType ord2 = ord_at(i + 2);
				if ((ord2 >= '0' && ord2 <= '9') || (ord2 >= 'A' && ord2 <= 'Z')) {
					char bytes[3] = { (char)ord1, (char)ord2, 0 };
					res += (char)strtol(bytes, nullptr, 16);
					i += 2;
				}
			} else {
				res += ord_at(i);
			}
		} else {
			res += ord_at(i);
		}
	}
	return String::utf8(res.ascii());
}

String String::uri_encode() const {
	const CharString temp = utf8();
	String res;
	for (int i = 0; i < temp.length(); ++i) {
		uint8_t ord = temp[i];
		if (ord == '.' || ord == '-' || ord == '~' || is_ascii_identifier_char(ord)) {
			res += ord;
		} else {
			char p[4] = { '%', 0, 0, 0 };
			static const char hex[16] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F' };
			p[1] = hex[ord >> 4];
			p[2] = hex[ord & 0xF];
			res += p;
		}
	}
	return res;
}

String String::uri_decode() const {
	CharString src = utf8();
	CharString res;
	for (int i = 0; i < src.length(); ++i) {
		if (src[i] == '%' && i + 2 < src.length()) {
			char ord1 = src[i + 1];
			if (is_digit(ord1) || is_ascii_upper_case(ord1)) {
				char ord2 = src[i + 2];
				if (is_digit(ord2) || is_ascii_upper_case(ord2)) {
					char bytes[3] = { (char)ord1, (char)ord2, 0 };
					res += (char)strtol(bytes, nullptr, 16);
					i += 2;
				}
			} else {
				res += src[i];
			}
		} else if (src[i] == '+') {
			res += ' ';
		} else {
			res += src[i];
		}
	}
	return String::utf8(res);
}

String String::c_escape() const {
	String escaped = *this;
	escaped = escaped.replace("\\", "\\\\");
	escaped = escaped.replace("\a", "\\a");
	escaped = escaped.replace("\b", "\\b");
	escaped = escaped.replace("\f", "\\f");
	escaped = escaped.replace("\n", "\\n");
	escaped = escaped.replace("\r", "\\r");
	escaped = escaped.replace("\t", "\\t");
	escaped = escaped.replace("\v", "\\v");
	escaped = escaped.replace("\'", "\\'");
	escaped = escaped.replace("\?", "\\?");
	escaped = escaped.replace("\"", "\\\"");

	return escaped;
}

String String::c_escape_multiline() const {
	String escaped = *this;
	escaped = escaped.replace("\\", "\\\\");
	escaped = escaped.replace("\"", "\\\"");

	return escaped;
}

String String::c_unescape() const {
	String escaped = *this;
	escaped = escaped.replace("\\a", "\a");
	escaped = escaped.replace("\\b", "\b");
	escaped = escaped.replace("\\f", "\f");
	escaped = escaped.replace("\\n", "\n");
	escaped = escaped.replace("\\r", "\r");
	escaped = escaped.replace("\\t", "\t");
	escaped = escaped.replace("\\v", "\v");
	escaped = escaped.replace("\\'", "\'");
	escaped = escaped.replace("\\\"", "\"");
	escaped = escaped.replace("\\?", "\?");
	escaped = escaped.replace("\\\\", "\\");

	return escaped;
}

String String::json_escape() const {
	String escaped = *this;
	escaped = escaped.replace("\\", "\\\\");
	escaped = escaped.replace("\b", "\\b");
	escaped = escaped.replace("\f", "\\f");
	escaped = escaped.replace("\n", "\\n");
	escaped = escaped.replace("\r", "\\r");
	escaped = escaped.replace("\t", "\\t");
	escaped = escaped.replace("\v", "\\v");
	escaped = escaped.replace("\"", "\\\"");

	return escaped;
}

String String::word_wrap(int p_chars_per_line) const {
	String ret;

	int line_start = 0;
	int line_end = 0; // End of last word on current line.
	int word_start = 0; // -1 if no word encountered. Leading spaces are part of a word.
	int word_length = 0;

	for (int i = 0; i < length(); i++) {
		const CharType c = operator[](i);

		switch (c) {
			case '\n': {
				// Force newline.
				ret += substr(line_start, i - line_start + 1);
				line_start = i + 1;
				line_end = line_start;
				word_start = line_start;
				word_length = 0;
			} break;

			case ' ':
			case '\t': {
				// A whitespace ends current word.
				if (word_length > 0) {
					line_end = i - 1;
					word_start = -1;
					word_length = 0;
				}
			} break;

			default: {
				if (word_start == -1) {
					word_start = i;
				}
				word_length += 1;

				if (word_length > p_chars_per_line) {
					// Word too long: wrap before current character.
					ret += substr(line_start, i - line_start) + "\n";
					line_start = i;
					line_end = i;
					word_start = i;
					word_length = 1;
				} else if (i - line_start + 1 > p_chars_per_line) {
					// Line too long: wrap after the last word.
					ret += substr(line_start, line_end - line_start + 1) + "\n";
					line_start = word_start;
					line_end = line_start;
				}
			} break;
		}
	}

	const int remaining = length() - line_start;
	if (remaining) {
		ret += substr(line_start, remaining);
	}

	return ret;
}

Error String::parse_url(String &r_scheme, String &r_host, int &r_port, String &r_path) const {
	// Splits the URL into scheme, host, port, path. Strip credentials when present.
	String base = *this;
	r_scheme = "";
	r_host = "";
	r_port = 0;
	r_path = "";
	int pos = base.find("://");
	// Scheme
	if (pos != -1) {
		r_scheme = base.substr(0, pos + 3).to_lower();
		base = base.substr(pos + 3, base.length() - pos - 3);
	}
	pos = base.find("/");
	// Path
	if (pos != -1) {
		r_path = base.substr(pos, base.length() - pos);
		base = base.substr(0, pos);
	}
	// Host
	pos = base.find("@");
	if (pos != -1) {
		// Strip credentials
		base = base.substr(pos + 1, base.length() - pos - 1);
	}
	if (base.begins_with("[")) {
		// Literal IPv6
		pos = base.rfind("]");
		if (pos == -1) {
			return ERR_INVALID_PARAMETER;
		}
		r_host = base.substr(1, pos - 1);
		base = base.substr(pos + 1, base.length() - pos - 1);
	} else {
		// Anything else
		if (base.get_slice_count(":") > 2) {
			return ERR_INVALID_PARAMETER;
		}
		pos = base.rfind(":");
		if (pos == -1) {
			r_host = base;
			base = "";
		} else {
			r_host = base.substr(0, pos);
			base = base.substr(pos, base.length() - pos);
		}
	}
	if (r_host.empty()) {
		return ERR_INVALID_PARAMETER;
	}
	r_host = r_host.to_lower();
	// Port
	if (base.begins_with(":")) {
		base = base.substr(1, base.length() - 1);
		if (!base.is_valid_integer()) {
			return ERR_INVALID_PARAMETER;
		}
		r_port = base.to_int();
		if (r_port < 1 || r_port > 65535) {
			return ERR_INVALID_PARAMETER;
		}
	}
	return OK;
}

String String::percent_encode() const {
	CharString cs = utf8();
	String encoded;
	for (int i = 0; i < cs.length(); i++) {
		uint8_t c = cs[i];
		if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '~' || c == '.') {
			char p[2] = { (char)c, 0 };
			encoded += p;
		} else {
			char p[4] = { '%', 0, 0, 0 };
			static const char hex[16] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' };

			p[1] = hex[c >> 4];
			p[2] = hex[c & 0xF];
			encoded += p;
		}
	}

	return encoded;
}
String String::percent_decode() const {
	CharString pe;

	CharString cs = utf8();
	for (int i = 0; i < cs.length(); i++) {
		uint8_t c = cs[i];
		if (c == '%' && i < length() - 2) {
			uint8_t a = LOWERCASE(cs[i + 1]);
			uint8_t b = LOWERCASE(cs[i + 2]);

			if (a >= '0' && a <= '9') {
				c = (a - '0') << 4;
			} else if (a >= 'a' && a <= 'f') {
				c = (a - 'a' + 10) << 4;
			} else {
				continue;
			}

			uint8_t d = 0;

			if (b >= '0' && b <= '9') {
				d = (b - '0');
			} else if (b >= 'a' && b <= 'f') {
				d = (b - 'a' + 10);
			} else {
				continue;
			}
			c += d;
			i += 2;
		}
		pe += c;
	}

	return String::utf8(pe.ptr());
}

String String::property_name_encode() const {
	// Escape and quote strings with extended ASCII or further Unicode characters
	// as well as '"', '=' or ' ' (32)
	const CharType *cstr = get_data();
	for (int i = 0; cstr[i]; i++) {
		if (cstr[i] == '=' || cstr[i] == '"' || cstr[i] == ';' || cstr[i] == '[' || cstr[i] == ']' || cstr[i] < 33 || cstr[i] > 126) {
			return "\"" + c_escape_multiline() + "\"";
		}
	}
	// Keep as is
	return *this;
}

// Changes made to the set of invalid characters must also be reflected in the String documentation.

static const char32_t invalid_node_name_characters[] = { '.', ':', '@', '/', '\"', '%', 0 };

String String::get_invalid_node_name_characters() {
	// Do not use this function for critical validation.
	String r;
	const char32_t *c = invalid_node_name_characters;
	while (*c) {
		if (c != invalid_node_name_characters) {
			r += " ";
		}
		r += String::chr(*c);
		c++;
	}
	return r;
}

String String::validate_node_name() const {
	// This is a critical validation in node addition, so it must be optimized.
	const char32_t *cn = ptr();
	if (cn == nullptr) {
		return String();
	}

	bool valid = true;
	uint32_t idx = 0;
	while (cn[idx]) {
		const char32_t *c = invalid_node_name_characters;
		while (*c) {
			if (cn[idx] == *c) {
				valid = false;
				break;
			}
			c++;
		}
		if (!valid) {
			break;
		}
		idx++;
	}

	if (valid) {
		return *this;
	}

	String validated = *this;
	char32_t *nn = validated.ptrw();
	while (nn[idx]) {
		const char32_t *c = invalid_node_name_characters;
		while (*c) {
			if (nn[idx] == *c) {
				nn[idx] = '_';
				break;
			}
			c++;
		}
		idx++;
	}

	return validated;
}

static _FORCE_INLINE_ bool _is_valid_identifier_bit(int p_index, char32_t p_char) {
	if (p_index == 0 && is_digit(p_char)) {
		return false; // No start with number plz.
	}
	return is_ascii_identifier_char(p_char);
}

String String::validate_identifier() const {
	if (empty()) {
		return "_"; // Empty string is not a valid identifier;
	}

	String result = *this;
	int len = result.length();
	char32_t *buffer = result.ptrw();

	for (int i = 0; i < len; i++) {
		if (!_is_valid_identifier_bit(i, buffer[i])) {
			buffer[i] = '_';
		}
	}

	return result;
}

bool String::is_valid_identifier() const {
	int len = length();

	if (len == 0) {
		return false;
	}

	const CharType *str = &operator[](0);

	for (int i = 0; i < len; i++) {
		if (i == 0) {
			if (str[0] >= '0' && str[0] <= '9') {
				return false; // no start with number plz
			}
		}

		bool valid_char = (str[i] >= '0' && str[i] <= '9') || (str[i] >= 'a' && str[i] <= 'z') || (str[i] >= 'A' && str[i] <= 'Z') || str[i] == '_';

		if (!valid_char) {
			return false;
		}
	}

	return true;
}

bool String::is_valid_integer() const {
	int len = length();

	if (len == 0) {
		return false;
	}

	int from = 0;
	if (len != 1 && (operator[](0) == '+' || operator[](0) == '-')) {
		from++;
	}

	for (int i = from; i < len; i++) {
		if (operator[](i) < '0' || operator[](i) > '9') {
			return false; // no start with number plz
		}
	}

	return true;
}

bool String::is_valid_hex_number(bool p_with_prefix) const {
	int len = length();

	if (len == 0) {
		return false;
	}

	int from = 0;
	if (len != 1 && (operator[](0) == '+' || operator[](0) == '-')) {
		from++;
	}

	if (p_with_prefix) {
		if (len < 3) {
			return false;
		}
		if (operator[](from) != '0' || operator[](from + 1) != 'x') {
			return false;
		}
		from += 2;
	}

	for (int i = from; i < len; i++) {
		CharType c = operator[](i);
		if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			continue;
		}
		return false;
	}

	return true;
};

bool String::is_valid_float() const {
	int len = length();

	if (len == 0) {
		return false;
	}

	int from = 0;
	if (operator[](0) == '+' || operator[](0) == '-') {
		from++;
	}

	bool exponent_found = false;
	bool period_found = false;
	bool sign_found = false;
	bool exponent_values_found = false;
	bool numbers_found = false;

	for (int i = from; i < len; i++) {
		if (operator[](i) >= '0' && operator[](i) <= '9') {
			if (exponent_found) {
				exponent_values_found = true;
			} else {
				numbers_found = true;
			}
		} else if (numbers_found && !exponent_found && operator[](i) == 'e') {
			exponent_found = true;
		} else if (!period_found && !exponent_found && operator[](i) == '.') {
			period_found = true;
		} else if ((operator[](i) == '-' || operator[](i) == '+') && exponent_found && !exponent_values_found && !sign_found) {
			sign_found = true;
		} else {
			return false; // no start with number plz
		}
	}

	return numbers_found;
}

bool String::is_valid_bool() const {
	int size = length();

	if (size == 1) {
		CharType c = ptr()[0];

		if (c == '0') {
			return true;
		} else if (c == '1') {
			return true;
		}

		return false;
	} else if (size == 4) {
		String l = to_lower();
		const CharType *p = l.ptr();

		if (p[0] == 't' && p[1] == 'r' && p[2] == 'u' && p[3] == 'e') {
			return true;
		} else {
			return false;
		}
	} else if (size == 5) {
		String l = to_lower();
		const CharType *p = l.ptr();

		if (p[0] == 'f' && p[1] == 'a' && p[2] == 'l' && p[3] == 's' && p[3] == 'e') {
			return true;
		} else {
			return false;
		}
	}

	return false;
}

bool String::is_valid_unsigned_integer() const {
	int len = length();

	if (len == 0) {
		return false;
	}

	int from = 0;
	if (len != 1 && (operator[](0) == '+')) {
		from++;
	}

	for (int i = from; i < len; i++) {
		if (operator[](i) < '0' || operator[](i) > '9') {
			return false; // no start with number plz
		}
	}

	return true;
}

bool String::is_valid_html_color() const {
	return Color::html_is_valid(*this);
}

bool String::is_valid_filename() const {
	String stripped = strip_edges();
	if (*this != stripped) {
		return false;
	}

	if (stripped == String()) {
		return false;
	}

	return !(find(":") != -1 || find("/") != -1 || find("\\") != -1 || find("?") != -1 || find("*") != -1 || find("\"") != -1 || find("|") != -1 || find("%") != -1 || find("<") != -1 || find(">") != -1);
}

bool String::is_valid_ip_address() const {
	if (find(":") >= 0) {
		Vector<String> ip = split(":");
		for (int i = 0; i < ip.size(); i++) {
			String n = ip[i];
			if (n.empty()) {
				continue;
			}
			if (n.is_valid_hex_number(false)) {
				int nint = n.hex_to_int(false);
				if (nint < 0 || nint > 0xffff) {
					return false;
				}
				continue;
			};
			if (!n.is_valid_ip_address()) {
				return false;
			}
		};

	} else {
		Vector<String> ip = split(".");
		if (ip.size() != 4) {
			return false;
		}
		for (int i = 0; i < ip.size(); i++) {
			String n = ip[i];
			if (!n.is_valid_integer()) {
				return false;
			}
			int val = n.to_int();
			if (val < 0 || val > 255) {
				return false;
			}
		}
	};

	return true;
}

Vector<uint8_t> String::to_ascii_buffer() const {
	const String *s = this;
	if (s->empty()) {
		return Vector<uint8_t>();
	}
	CharString charstr = s->ascii();

	Vector<uint8_t> retval;
	size_t len = charstr.length();
	retval.resize(len);
	uint8_t *w = retval.ptrw();
	memcpy(w, charstr.ptr(), len);

	return retval;
}

Vector<uint8_t> String::to_utf8_buffer() const {
	const String *s = this;
	if (s->empty()) {
		return Vector<uint8_t>();
	}
	CharString charstr = s->utf8();

	Vector<uint8_t> retval;
	size_t len = charstr.length();
	retval.resize(len);
	uint8_t *w = retval.ptrw();
	memcpy(w, charstr.ptr(), len);

	return retval;
}

Vector<uint8_t> String::to_utf16_buffer() const {
	const String *s = this;
	if (s->empty()) {
		return Vector<uint8_t>();
	}
	Char16String charstr = s->utf16();

	Vector<uint8_t> retval;
	size_t len = charstr.length() * sizeof(char16_t);
	retval.resize(len);
	uint8_t *w = retval.ptrw();
	memcpy(w, (const void *)charstr.ptr(), len);

	return retval;
}

Vector<uint8_t> String::to_utf32_buffer() const {
	const String *s = this;
	if (s->empty()) {
		return Vector<uint8_t>();
	}

	Vector<uint8_t> retval;
	size_t len = s->length() * sizeof(CharType);
	retval.resize(len);
	uint8_t *w = retval.ptrw();
	memcpy(w, (const void *)s->ptr(), len);

	return retval;
}

String::String(const char *p_str) {
	copy_from(p_str);
}

String::String(const wchar_t *p_str) {
	copy_from(p_str);
}

String::String(const CharType *p_str) {
	copy_from(p_str);
}

String::String(const char *p_str, int p_clip_to_len) {
	copy_from(p_str, p_clip_to_len);
}

String::String(const wchar_t *p_str, int p_clip_to_len) {
	copy_from(p_str, p_clip_to_len);
}

String::String(const CharType *p_str, int p_clip_to_len) {
	copy_from(p_str, p_clip_to_len);
}

String::String(const StrRange &p_range) {
	if (!p_range.c_str) {
		return;
	}
	copy_from(p_range.c_str, p_range.len);
}

// Other conversions

int String::hex_to_int(bool p_with_prefix) const {
	int len = length();
	ERR_FAIL_COND_V_MSG(p_with_prefix ? len < 3 : len == 0, 0, String("Invalid hexadecimal notation length in string ") + (p_with_prefix ? "with" : "without") + " prefix \"" + *this + "\".");

	const CharType *s = ptr();

	int sign = s[0] == '-' ? -1 : 1;

	if (sign < 0) {
		s++;
	}

	if (p_with_prefix) {
		ERR_FAIL_COND_V_MSG(s[0] != '0' || LOWERCASE(s[1]) != 'x', 0, "Invalid hexadecimal notation prefix in string \"" + *this + "\".");
		s += 2;
	}

	int hex = 0;

	while (*s) {
		CharType c = LOWERCASE(*s);
		int n;
		if (c >= '0' && c <= '9') {
			n = c - '0';
		} else if (c >= 'a' && c <= 'f') {
			n = (c - 'a') + 10;
		} else {
			ERR_FAIL_V_MSG(0, "Invalid hexadecimal notation character \"" + chr(*s) + "\" in string \"" + *this + "\".");
		}
		// Check for overflow/underflow, with special case to ensure INT32_MIN does not result in error
		bool overflow = ((hex > INT32_MAX / 16) && (sign == 1 || (sign == -1 && hex != (INT32_MAX >> 4) + 1))) || (sign == -1 && hex == (INT32_MAX >> 4) + 1 && c > '0');
		ERR_FAIL_COND_V_MSG(overflow, sign == 1 ? INT32_MAX : INT32_MIN, "Cannot represent " + *this + " as a 32-bit signed integer, since the value is " + (sign == 1 ? "too large." : "too small."));
		hex *= 16;
		hex += n;
		s++;
	}

	return hex * sign;
}

int64_t String::hex_to_int64(bool p_with_prefix) const {
	int len = length();
	ERR_FAIL_COND_V_MSG(p_with_prefix ? len < 3 : len == 0, 0, String("Invalid hexadecimal notation length in string ") + (p_with_prefix ? "with" : "without") + " prefix \"" + *this + "\".");

	const CharType *s = ptr();

	int64_t sign = s[0] == '-' ? -1 : 1;

	if (sign < 0) {
		s++;
	}

	if (p_with_prefix) {
		ERR_FAIL_COND_V_MSG(s[0] != '0' || LOWERCASE(s[1]) != 'x', 0, "Invalid hexadecimal notation prefix in string \"" + *this + "\".");
		s += 2;
	}

	int64_t hex = 0;

	while (*s) {
		CharType c = LOWERCASE(*s);
		int64_t n;
		if (c >= '0' && c <= '9') {
			n = c - '0';
		} else if (c >= 'a' && c <= 'f') {
			n = (c - 'a') + 10;
		} else {
			ERR_FAIL_V_MSG(0, "Invalid hexadecimal notation character \"" + chr(*s) + "\" in string \"" + *this + "\".");
		}
		bool overflow = ((hex > INT64_MAX / 16) && (sign == 1 || (sign == -1 && hex != (INT64_MAX >> 4) + 1))) || (sign == -1 && hex == (INT64_MAX >> 4) + 1 && c > '0');
		ERR_FAIL_COND_V_MSG(overflow, sign == 1 ? INT64_MAX : INT64_MIN, "Cannot represent " + *this + " as a 64-bit signed integer, since the value is " + (sign == 1 ? "too large." : "too small."));
		hex *= 16;
		hex += n;
		s++;
	}

	return hex * sign;
}

int64_t String::bin_to_int64(bool p_with_prefix) const {
	int len = length();
	ERR_FAIL_COND_V_MSG(p_with_prefix ? len < 3 : len == 0, 0, String("Invalid binary notation length in string ") + (p_with_prefix ? "with" : "without") + " prefix \"" + *this + "\".");

	const CharType *s = ptr();

	int64_t sign = s[0] == '-' ? -1 : 1;

	if (sign < 0) {
		s++;
	}

	if (p_with_prefix) {
		ERR_FAIL_COND_V_MSG(s[0] != '0' || LOWERCASE(s[1]) != 'b', 0, "Invalid binary notation prefix in string \"" + *this + "\".");
		s += 2;
	}

	int64_t binary = 0;

	while (*s) {
		CharType c = LOWERCASE(*s);
		int64_t n;
		if (c == '0' || c == '1') {
			n = c - '0';
		} else {
			ERR_FAIL_V_MSG(0, "Invalid binary notation character \"" + chr(*s) + "\" in string \"" + *this + "\".");
		}
		// Check for overflow/underflow, with special case to ensure INT64_MIN does not result in error
		bool overflow = ((binary > INT64_MAX / 2) && (sign == 1 || (sign == -1 && binary != (INT64_MAX >> 1) + 1))) || (sign == -1 && binary == (INT64_MAX >> 1) + 1 && c > '0');
		ERR_FAIL_COND_V_MSG(overflow, sign == 1 ? INT64_MAX : INT64_MIN, "Cannot represent " + *this + " as a 64-bit signed integer, since the value is " + (sign == 1 ? "too large." : "too small."));
		binary *= 2;
		binary += n;
		s++;
	}

	return binary * sign;
}

template <class C>
static double built_in_strtod(
		/* A decimal ASCII floating-point number,
		 * optionally preceded by white space. Must
		 * have form "-I.FE-X", where I is the integer
		 * part of the mantissa, F is the fractional
		 * part of the mantissa, and X is the
		 * exponent. Either of the signs may be "+",
		 * "-", or omitted. Either I or F may be
		 * omitted, or both. The decimal point isn't
		 * necessary unless F is present. The "E" may
		 * actually be an "e". E and X may both be
		 * omitted (but not just one). */
		const C *string,
		/* If non-nullptr, store terminating Cacter's
		 * address here. */
		C **endPtr = nullptr) {
	/* Largest possible base 10 exponent. Any
	 * exponent larger than this will already
	 * produce underflow or overflow, so there's
	 * no need to worry about additional digits. */
	static const int maxExponent = 511;
	/* Table giving binary powers of 10. Entry
	 * is 10^2^i. Used to convert decimal
	 * exponents into floating-point numbers. */
	static const double powersOf10[] = {
		10.,
		100.,
		1.0e4,
		1.0e8,
		1.0e16,
		1.0e32,
		1.0e64,
		1.0e128,
		1.0e256
	};

	bool sign, expSign = false;
	double fraction, dblExp;
	const double *d;
	const C *p;
	int c;
	/* Exponent read from "EX" field. */
	int exp = 0;
	/* Exponent that derives from the fractional
	 * part. Under normal circumstances, it is
	 * the negative of the number of digits in F.
	 * However, if I is very long, the last digits
	 * of I get dropped (otherwise a long I with a
	 * large negative exponent could cause an
	 * unnecessary overflow on I alone). In this
	 * case, fracExp is incremented one for each
	 * dropped digit. */
	int fracExp = 0;
	/* Number of digits in mantissa. */
	int mantSize;
	/* Number of mantissa digits BEFORE decimal point. */
	int decPt;
	/* Temporarily holds location of exponent in string. */
	const C *pExp;

	/*
	 * Strip off leading blanks and check for a sign.
	 */

	p = string;
	while (*p == ' ' || *p == '\t' || *p == '\n') {
		p += 1;
	}
	if (*p == '-') {
		sign = true;
		p += 1;
	} else {
		if (*p == '+') {
			p += 1;
		}
		sign = false;
	}

	/*
	 * Count the number of digits in the mantissa (including the decimal
	 * point), and also locate the decimal point.
	 */

	decPt = -1;
	for (mantSize = 0;; mantSize += 1) {
		c = *p;
		if (!IS_DIGIT(c)) {
			if ((c != '.') || (decPt >= 0)) {
				break;
			}
			decPt = mantSize;
		}
		p += 1;
	}

	/*
	 * Now suck up the digits in the mantissa. Use two integers to collect 9
	 * digits each (this is faster than using floating-point). If the mantissa
	 * has more than 18 digits, ignore the extras, since they can't affect the
	 * value anyway.
	 */

	pExp = p;
	p -= mantSize;
	if (decPt < 0) {
		decPt = mantSize;
	} else {
		mantSize -= 1; /* One of the digits was the point. */
	}
	if (mantSize > 18) {
		fracExp = decPt - 18;
		mantSize = 18;
	} else {
		fracExp = decPt - mantSize;
	}
	if (mantSize == 0) {
		fraction = 0.0;
		p = string;
		goto done;
	} else {
		int frac1, frac2;

		frac1 = 0;
		for (; mantSize > 9; mantSize -= 1) {
			c = *p;
			p += 1;
			if (c == '.') {
				c = *p;
				p += 1;
			}
			frac1 = 10 * frac1 + (c - '0');
		}
		frac2 = 0;
		for (; mantSize > 0; mantSize -= 1) {
			c = *p;
			p += 1;
			if (c == '.') {
				c = *p;
				p += 1;
			}
			frac2 = 10 * frac2 + (c - '0');
		}
		fraction = (1.0e9 * frac1) + frac2;
	}

	/*
	 * Skim off the exponent.
	 */

	p = pExp;
	if ((*p == 'E') || (*p == 'e')) {
		p += 1;
		if (*p == '-') {
			expSign = true;
			p += 1;
		} else {
			if (*p == '+') {
				p += 1;
			}
			expSign = false;
		}
		if (!IS_DIGIT(CharType(*p))) {
			p = pExp;
			goto done;
		}
		while (IS_DIGIT(CharType(*p))) {
			exp = exp * 10 + (*p - '0');
			p += 1;
		}
	}
	if (expSign) {
		exp = fracExp - exp;
	} else {
		exp = fracExp + exp;
	}

	/*
	 * Generate a floating-point number that represents the exponent. Do this
	 * by processing the exponent one bit at a time to combine many powers of
	 * 2 of 10. Then combine the exponent with the fraction.
	 */

	if (exp < 0) {
		expSign = true;
		exp = -exp;
	} else {
		expSign = false;
	}

	if (exp > maxExponent) {
		exp = maxExponent;
		//LOG_WARN("Exponent too high");
	}
	dblExp = 1.0;
	for (d = powersOf10; exp != 0; exp >>= 1, ++d) {
		if (exp & 01) {
			dblExp *= *d;
		}
	}
	if (expSign) {
		fraction /= dblExp;
	} else {
		fraction *= dblExp;
	}

done:
	if (endPtr != nullptr) {
		*endPtr = (C *)p;
	}

	if (sign) {
		return -fraction;
	}
	return fraction;
}

double String::to_double() const {
	if (empty()) {
		return 0;
	}

	return built_in_strtod<CharType>(get_data());
}

float String::to_float() const {
	if (empty()) {
		return 0;
	}

	return built_in_strtod<CharType>(get_data());
}

int String::to_int() const {
	if (length() == 0) {
		return 0;
	}

	int to = (find(".") >= 0) ? find(".") : length();

	int integer = 0;
	int sign = 1;

	for (int i = 0; i < to; i++) {
		CharType c = operator[](i);
		if (c >= '0' && c <= '9') {
			bool overflow = (integer > INT32_MAX / 10) || (integer == INT32_MAX / 10 && ((sign == 1 && c > '7') || (sign == -1 && c > '8')));
			ERR_FAIL_COND_V_MSG(overflow, sign == 1 ? INT32_MAX : INT32_MIN, "Cannot represent " + *this + " as a 32-bit signed integer, since the value is " + (sign == 1 ? "too large." : "too small."));
			integer *= 10;
			integer += c - '0';

		} else if (integer == 0 && c == '-') {
			sign = -sign;
		}
	}

	return integer * sign;
}

bool String::to_bool() const {
	if (length() == 0) {
		return false;
	}

	if (is_numeric()) {
		return to_int() != 0;
	}

	return to_lower() == "true";
}

uint32_t String::to_uint() const {
	if (is_numeric()) {
		return static_cast<uint32_t>(to_int());
	}

	return 0;
}

int64_t String::to_int64() const {
	if (length() == 0) {
		return 0;
	}

	int to = (find(".") >= 0) ? find(".") : length();

	int64_t integer = 0;
	int64_t sign = 1;

	for (int i = 0; i < to; i++) {
		CharType c = operator[](i);
		if (c >= '0' && c <= '9') {
			bool overflow = (integer > INT64_MAX / 10) || (integer == INT64_MAX / 10 && ((sign == 1 && c > '7') || (sign == -1 && c > '8')));
			ERR_FAIL_COND_V_MSG(overflow, sign == 1 ? INT64_MAX : INT64_MIN, "Cannot represent " + *this + " as a 64-bit signed integer, since the value is " + (sign == 1 ? "too large." : "too small."));
			integer *= 10;
			integer += c - '0';

		} else if (integer == 0 && c == '-') {
			sign = -sign;
		}
	}

	return integer * sign;
}

int64_t String::to_int(const char *p_str, int p_len) {
	int to = 0;
	if (p_len >= 0) {
		to = p_len;
	} else {
		while (p_str[to] != 0 && p_str[to] != '.') {
			to++;
		}
	}

	int64_t integer = 0;
	int64_t sign = 1;

	for (int i = 0; i < to; i++) {
		char c = p_str[i];
		if (is_digit(c)) {
			bool overflow = (integer > INT64_MAX / 10) || (integer == INT64_MAX / 10 && ((sign == 1 && c > '7') || (sign == -1 && c > '8')));
			ERR_FAIL_COND_V_MSG(overflow, sign == 1 ? INT64_MAX : INT64_MIN, "Cannot represent " + String(p_str).substr(0, to) + " as a 64-bit signed integer, since the value is " + (sign == 1 ? "too large." : "too small."));
			integer *= 10;
			integer += c - '0';

		} else if (c == '-' && integer == 0) {
			sign = -sign;
		} else if (c != ' ') {
			break;
		}
	}

	return integer * sign;
}

int64_t String::to_int(const wchar_t *p_str, int p_len) {
	int to = 0;
	if (p_len >= 0) {
		to = p_len;
	} else {
		while (p_str[to] != 0 && p_str[to] != '.') {
			to++;
		}
	}

	int64_t integer = 0;
	int64_t sign = 1;

	for (int i = 0; i < to; i++) {
		wchar_t c = p_str[i];
		if (is_digit(c)) {
			bool overflow = (integer > INT64_MAX / 10) || (integer == INT64_MAX / 10 && ((sign == 1 && c > '7') || (sign == -1 && c > '8')));
			ERR_FAIL_COND_V_MSG(overflow, sign == 1 ? INT64_MAX : INT64_MIN, "Cannot represent " + String(p_str).substr(0, to) + " as a 64-bit signed integer, since the value is " + (sign == 1 ? "too large." : "too small."));
			integer *= 10;
			integer += c - '0';

		} else if (c == '-' && integer == 0) {
			sign = -sign;
		} else if (c != ' ') {
			break;
		}
	}

	return integer * sign;
}

int64_t String::to_int(const CharType *p_str, int p_len, bool p_clamp) {
	if (p_len == 0 || !p_str[0]) {
		return 0;
	}
	///@todo make more exact so saving and loading does not lose precision

	int64_t integer = 0;
	int64_t sign = 1;
	int reading = READING_SIGN;

	const CharType *str = p_str;
	const CharType *limit = &p_str[p_len];

	while (*str && reading != READING_DONE && str != limit) {
		CharType c = *(str++);
		switch (reading) {
			case READING_SIGN: {
				if (is_digit(c)) {
					reading = READING_INT;
					// let it fallthrough
				} else if (c == '-') {
					sign = -1;
					reading = READING_INT;
					break;
				} else if (c == '+') {
					sign = 1;
					reading = READING_INT;
					break;
				} else {
					break;
				}
				FALLTHROUGH;
			}
			case READING_INT: {
				if (is_digit(c)) {
					if (integer > INT64_MAX / 10) {
						String number("");
						str = p_str;
						while (*str && str != limit) {
							number += *(str++);
						}
						if (p_clamp) {
							if (sign == 1) {
								return INT64_MAX;
							} else {
								return INT64_MIN;
							}
						} else {
							ERR_FAIL_V_MSG(sign == 1 ? INT64_MAX : INT64_MIN, "Cannot represent " + number + " as a 64-bit signed integer, since the value is " + (sign == 1 ? "too large." : "too small."));
						}
					}
					integer *= 10;
					integer += c - '0';
				} else {
					reading = READING_DONE;
				}

			} break;
		}
	}

	return sign * integer;
}

double String::to_float(const char *p_str) {
	return built_in_strtod<char>(p_str);
}

double String::to_float(const wchar_t *p_str, const wchar_t **r_end) {
	return built_in_strtod<wchar_t>(p_str, (wchar_t **)r_end);
}
double String::to_float(const CharType *p_str, const CharType **r_end) {
	return built_in_strtod<CharType>(p_str, (CharType **)r_end);
}

double String::to_double(const char *p_str) {
	return built_in_strtod<char>(p_str);
}

double String::to_double(const wchar_t *p_str, const wchar_t **r_end) {
	return built_in_strtod<wchar_t>(p_str, (wchar_t **)r_end);
}
double String::to_double(const CharType *p_str, const CharType **r_end) {
	return built_in_strtod<CharType>(p_str, (CharType **)r_end);
}

uint32_t String::num_characters(int64_t p_int) {
	int r = 1;
	if (p_int < 0) {
		r += 1;
		if (p_int == INT64_MIN) {
			p_int = INT64_MAX;
		} else {
			p_int = -p_int;
		}
	}
	while (p_int >= 10) {
		p_int /= 10;
		r++;
	}
	return r;
}

bool String::_base_is_subsequence_of(const String &p_string, bool case_insensitive) const {
	int len = length();
	if (len == 0) {
		// Technically an empty string is subsequence of any string
		return true;
	}

	if (len > p_string.length()) {
		return false;
	}

	const CharType *src = &operator[](0);
	const CharType *tgt = &p_string[0];

	for (; *src && *tgt; tgt++) {
		bool match = false;
		if (case_insensitive) {
			CharType srcc = _find_lower(*src);
			CharType tgtc = _find_lower(*tgt);
			match = srcc == tgtc;
		} else {
			match = *src == *tgt;
		}
		if (match) {
			src++;
			if (!*src) {
				return true;
			}
		}
	}

	return false;
}

bool operator==(const char *p_chr, const String &p_str) {
	return p_str == p_chr;
}

bool operator==(const wchar_t *p_chr, const String &p_str) {
#ifdef WINDOWS_ENABLED
	// wchar_t is 16-bit
	return p_str == String::utf16((const char16_t *)p_chr);
#else
	// wchar_t is 32-bi
	return p_str == String((const CharType *)p_chr);
#endif
}

bool operator!=(const char *p_chr, const String &p_str) {
	return !(p_str == p_chr);
}

bool operator!=(const wchar_t *p_chr, const String &p_str) {
#ifdef WINDOWS_ENABLED
	// wchar_t is 16-bit
	return !(p_str == String::utf16((const char16_t *)p_chr));
#else
	// wchar_t is 32-bi
	return !(p_str == String((const CharType *)p_chr));
#endif
}

String operator+(const char *p_chr, const String &p_str) {
	String tmp = p_chr;
	tmp += p_str;
	return tmp;
}

String operator+(const wchar_t *p_chr, const String &p_str) {
#ifdef WINDOWS_ENABLED
	// wchar_t is 16-bit
	String tmp = String::utf16((const char16_t *)p_chr);
#else
	// wchar_t is 32-bit
	String tmp = (const CharType *)p_chr;
#endif
	tmp += p_str;
	return tmp;
}

String operator+(CharType p_chr, const String &p_str) {
	return (String::chr(p_chr) + p_str);
}

String itos(int64_t p_val) {
	return String::num_int64(p_val);
}

String uitos(uint64_t p_val) {
	return String::num_uint64(p_val);
}

String rtos(double p_val) {
	return String::num(p_val);
}

String rtoss(double p_val) {
	return String::num_scientific(p_val);
}

#ifdef TOOLS_ENABLED
String TTR(const String &p_text, const String &p_context) {
	return p_text;
}

/* DTR is used for the documentation, handling descriptions extracted from the XML.
 * It also replaces `$DOCS_URL` with the actual URL to the documentation's branch,
 * to allow dehardcoding it in the XML and doing proper substitutions everywhere.
 */
String DTR(const String &p_text) {
	// Comes straight from the XML, so remove indentation and any trailing whitespace.
	const String text = p_text.dedent().strip_edges();

	return text.replace("$DOCS_URL", VERSION_DOCS_URL);
}
#endif

String RTR(const String &p_text) {
	return p_text;
}
#line 0

#line 1 "sfw/core/logger.cpp"

void RLogger::print_trace(const String &str) {
	print_trace(str.utf8().get_data());
}
void RLogger::print_trace(const char *str) {
	printf("T %s\n", str);
}
void RLogger::print_trace(const char *p_function, const char *p_file, int p_line, const char *str) {
	printf("T | %s::%s:%d | %s\n", p_file, p_function, p_line, str);
}
void RLogger::print_trace(const char *p_function, const char *p_file, int p_line, const String &str) {
	printf("T | %s::%s:%d | %s\n", p_file, p_function, p_line, str.utf8().get_data());
}

void RLogger::print_message(const String &str) {
	print_message(str.utf8().get_data());
}
void RLogger::print_message(const char *str) {
	printf("M %s\n", str);
}
void RLogger::print_message(const char *p_function, const char *p_file, int p_line, const char *str) {
	printf("M | %s::%s:%d | %s\n", p_file, p_function, p_line, str);
}
void RLogger::print_message(const char *p_function, const char *p_file, int p_line, const String &str) {
	printf("M | %s::%s:%d | %s\n", p_file, p_function, p_line, str.utf8().get_data());
}

void RLogger::print_warning(const String &str) {
	print_warning(str.utf8().get_data());
}
void RLogger::print_warning(const char *str) {
	printf("W %s\n", str);
}
void RLogger::print_warning(const char *p_function, const char *p_file, int p_line, const char *str) {
	printf("W | %s::%s:%d | %s\n", p_file, p_function, p_line, str);
}
void RLogger::print_warning(const char *p_function, const char *p_file, int p_line, const String &str) {
	printf("W | %s::%s:%d | %s\n", p_file, p_function, p_line, str.utf8().get_data());
}

void RLogger::print_error(const String &str) {
	print_error(str.utf8().get_data());
}
void RLogger::print_error(const char *str) {
	printf("E %s\n", str);
}

void RLogger::print_error(const char *p_function, const char *p_file, int p_line, const char *str) {
	printf("E | %s::%s:%d | %s\n", p_file, p_function, p_line, str);
}
void RLogger::print_error(const char *p_function, const char *p_file, int p_line, const String &str) {
	printf("E | %s::%s:%d | %s\n", p_file, p_function, p_line, str.utf8().get_data());
}
void RLogger::print_msg_error(const char *p_function, const char *p_file, int p_line, const char *p_msg, const char *str) {
	printf("E | %s::%s:%d | :: %s. %s\n", p_file, p_function, p_line, str, p_msg);
}
void RLogger::print_index_error(const char *p_function, const char *p_file, int p_line, const int index, const int size, const char *str) {
	printf("E (INDEX) | %s::%s:%d | :: index: %d/%d. %s\n", p_file, p_function, p_line, index, size, str);
}

void RLogger::log_trace(const String &str) {
	log_trace(str.utf8().get_data());
}
void RLogger::log_trace(const char *str) {
	printf("T %s\n", str);
}
void RLogger::log_trace(const char *p_function, const char *p_file, int p_line, const char *str) {
	printf("T | %s::%s:%d | %s\n", p_file, p_function, p_line, str);
}
void RLogger::log_trace(const char *p_function, const char *p_file, int p_line, const String &str) {
	printf("T | %s::%s:%d | %s\n", p_file, p_function, p_line, str.utf8().get_data());
}

void RLogger::log_message(const String &str) {
	log_message(str.utf8().get_data());
}
void RLogger::log_message(const char *str) {
	printf("M %s\n", str);
}
void RLogger::log_message(const char *p_function, const char *p_file, int p_line, const char *str) {
	printf("M | %s::%s:%d | %s\n", p_file, p_function, p_line, str);
}
void RLogger::log_message(const char *p_function, const char *p_file, int p_line, const String &str) {
	printf("M | %s::%s:%d | %s\n", p_file, p_function, p_line, str.utf8().get_data());
}

void RLogger::log_warning(const String &str) {
	log_warning(str.utf8().get_data());
}
void RLogger::log_warning(const char *str) {
	printf("W %s\n", str);
}
void RLogger::log_warning(const char *p_function, const char *p_file, int p_line, const char *str) {
	printf("W | %s::%s:%d | %s\n", p_file, p_function, p_line, str);
}
void RLogger::log_warning(const char *p_function, const char *p_file, int p_line, const String &str) {
	printf("W | %s::%s:%d | %s\n", p_file, p_function, p_line, str.utf8().get_data());
}

void RLogger::log_error(const String &str) {
	log_error(str.utf8().get_data());
}
void RLogger::log_error(const char *str) {
	printf("E %s\n", str);
}

void RLogger::log_error(const char *p_function, const char *p_file, int p_line, const char *str) {
	printf("E | %s::%s:%d | %s\n", p_file, p_function, p_line, str);
}
void RLogger::log_error(const char *p_function, const char *p_file, int p_line, const String &str) {
	printf("E | %s::%s:%d | %s\n", p_file, p_function, p_line, str.utf8().get_data());
}
void RLogger::log_msg_error(const char *p_function, const char *p_file, int p_line, const char *p_msg, const char *str) {
	printf("E | %s::%s:%d | :: %s. %s\n", p_file, p_function, p_line, str, p_msg);
}
void RLogger::log_index_error(const char *p_function, const char *p_file, int p_line, const int index, const int size, const char *str) {
	printf("E (INDEX) | %s::%s:%d | :: index: %d/%d. %s\n", p_file, p_function, p_line, index, size, str);
}
void RLogger::log_index_error(const char *p_function, const char *p_file, int p_line, const int index, const int size, const String &str) {
	printf("E (INDEX) | %s::%s:%d | :: index: %d/%d. %s\n", p_file, p_function, p_line, index, size, str.utf8().get_data());
}

String *RLogger::get_string_ptr(const int p_default_size) {
	return memnew(String());
}
String *RLogger::get_string_ptr(const char *p_function, const char *p_file, int p_line, const int p_default_size) {
	String *s = memnew(String());

	s->operator+=(p_function);
	s->operator+=("::");
	s->operator+=(p_file);
	s->operator+=(":");
	s->operator+=(String::num(p_line));
	s->operator+=(" | ");

	return s;
}
String *RLogger::get_string_ptr(const char *p_prefix, const char *p_function, const char *p_file, int p_line, const int p_default_size) {
	String *s = memnew(String());

	s->operator+=(p_prefix);
	s->operator+=(" | ");
	s->operator+=(p_function);
	s->operator+=("::");
	s->operator+=(p_file);
	s->operator+=(":");
	s->operator+=(String::num(p_line));
	s->operator+=(" | ");

	return s;
}
void RLogger::return_string_ptr(String *str) {
	memdelete(str);
}

String *RLogger::get_trace_string_ptr(const int p_default_size) {
	String *str = get_string_ptr(p_default_size);
	str->operator+=("T ");
	return str;
}
String *RLogger::get_message_string_ptr(const int p_default_size) {
	String *str = get_string_ptr(p_default_size);
	str->operator+=("M ");
	return str;
}
String *RLogger::get_warning_string_ptr(const int p_default_size) {
	String *str = get_string_ptr(p_default_size);
	str->operator+=("W ");
	return str;
}
String *RLogger::get_error_string_ptr(const int p_default_size) {
	String *str = get_string_ptr(p_default_size);
	str->operator+=("E ");
	return str;
}

String *RLogger::get_trace_string_ptr(const char *p_function, const char *p_file, int p_line, const int p_default_size) {
	return get_string_ptr("T", p_function, p_file, p_line, p_default_size);
}
String *RLogger::get_message_string_ptr(const char *p_function, const char *p_file, int p_line, const int p_default_size) {
	return get_string_ptr("M", p_function, p_file, p_line, p_default_size);
}
String *RLogger::get_warning_string_ptr(const char *p_function, const char *p_file, int p_line, const int p_default_size) {
	return get_string_ptr("W", p_function, p_file, p_line, p_default_size);
}
String *RLogger::get_error_string_ptr(const char *p_function, const char *p_file, int p_line, const int p_default_size) {
	return get_string_ptr("E", p_function, p_file, p_line, p_default_size);
}

void RLogger::log_ptr(String *str) {
	printf("%s\n", str->utf8().get_data());
}

void RLogger::log_ret_ptr(String *str) {
	log_ptr(str);

	return_string_ptr(str);
}
#line 0

#line 1 "sfw/core/string_name.cpp"
/*************************************************************************/
/*  string_name.cpp                                                      */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

StaticCString StaticCString::create(const char *p_ptr) {
	StaticCString scs;
	scs.ptr = p_ptr;
	return scs;
}

StringName::_Data *StringName::_table[STRING_TABLE_LEN];

StringName _scs_create(const char *p_chr, bool p_static) {
	return (p_chr[0] ? StringName(StaticCString::create(p_chr), p_static) : StringName());
}

bool StringName::configured = false;
Mutex StringName::lock;

#ifdef DEBUG_ENABLED
bool StringName::debug_stringname = false;
#endif

void StringName::setup() {
	ERR_FAIL_COND(configured);
	for (int i = 0; i < STRING_TABLE_LEN; i++) {
		_table[i] = nullptr;
	}
	configured = true;
}

void StringName::cleanup() {
	lock.lock();

#ifdef DEBUG_ENABLED
	if (unlikely(debug_stringname)) {
		Vector<_Data *> data;
		for (int i = 0; i < STRING_TABLE_LEN; i++) {
			_Data *d = _table[i];
			while (d) {
				data.push_back(d);
				d = d->next;
			}
		}

		LOG_MSG("\nStringName reference ranking (from most to least referenced):\n");

		data.sort_custom<DebugSortReferences>();
		int unreferenced_stringnames = 0;
		int rarely_referenced_stringnames = 0;
		for (int i = 0; i < data.size(); i++) {
			LOG_MSG(itos(i + 1) + ": " + data[i]->get_name() + " - " + itos(data[i]->debug_references));
			if (data[i]->debug_references == 0) {
				unreferenced_stringnames += 1;
			} else if (data[i]->debug_references < 5) {
				rarely_referenced_stringnames += 1;
			}
		}

		LOG_MSG(vformat("\nOut of %d StringNames, %d StringNames were never referenced during this run (0 times) (%.2f%%).", data.size(), unreferenced_stringnames, unreferenced_stringnames / float(data.size()) * 100));
		LOG_MSG(vformat("Out of %d StringNames, %d StringNames were rarely referenced during this run (1-4 times) (%.2f%%).", data.size(), rarely_referenced_stringnames, rarely_referenced_stringnames / float(data.size()) * 100));
	}
#endif

	//int lost_strings = 0;
	for (int i = 0; i < STRING_TABLE_LEN; i++) {
		while (_table[i]) {
			_Data *d = _table[i];

			/*
			if (d->static_count.get() != d->refcount.get()) {
				lost_strings++;
				if (OS::get_singleton()->is_stdout_verbose()) {
					if (d->cname) {
						LOG_MSG("Orphan StringName: " + String(d->cname));
					} else {
						LOG_MSG("Orphan StringName: " + String(d->name));
					}
				}
			}
			*/

			_table[i] = _table[i]->next;
			memdelete(d);
		}
	}

	/*
	if (lost_strings) {
		print_verbose("StringName: " + itos(lost_strings) + " unclaimed string names at exit.");
	}
	*/

	configured = false;

	lock.unlock();
}

void StringName::unref() {
	ERR_FAIL_COND(!configured);

	if (_data && _data->refcount.unref()) {
		lock.lock();

		if (_data->static_count.get() > 0) {
			if (_data->cname) {
				ERR_PRINT("BUG: Unreferenced static string to 0: " + String(_data->cname));
			} else {
				ERR_PRINT("BUG: Unreferenced static string to 0: " + String(_data->name));
			}
		}

		if (_data->prev) {
			_data->prev->next = _data->next;
		} else {
			if (_table[_data->idx] != _data) {
				ERR_PRINT("BUG!");
			}

			_table[_data->idx] = _data->next;
		}

		if (_data->next) {
			_data->next->prev = _data->prev;
		}

		memdelete(_data);
		lock.unlock();
	}

	_data = nullptr;
}

bool StringName::operator==(const String &p_name) const {
	if (!_data) {
		return (p_name.length() == 0);
	}

	return (_data->get_name() == p_name);
}

bool StringName::operator==(const char *p_name) const {
	if (!_data) {
		return (p_name[0] == 0);
	}

	return (_data->get_name() == p_name);
}

bool StringName::operator!=(const String &p_name) const {
	return !(operator==(p_name));
}

bool StringName::operator!=(const StringName &p_name) const {
	// the real magic of all this mess happens here.
	// this is why path comparisons are very fast
	return _data != p_name._data;
}

void StringName::operator=(const StringName &p_name) {
	if (this == &p_name) {
		return;
	}

	unref();

	if (p_name._data && p_name._data->refcount.ref()) {
		_data = p_name._data;
	}
}

StringName::StringName(const StringName &p_name) {
	_data = nullptr;

	ERR_FAIL_COND(!configured);

	if (p_name._data && p_name._data->refcount.ref()) {
		_data = p_name._data;
	}
}

StringName::StringName(const char *p_name, bool p_static) {
	_data = nullptr;

	ERR_FAIL_COND(!configured);

	if (!p_name || p_name[0] == 0) {
		return; //empty, ignore
	}

	lock.lock();

	uint32_t hash = String::hash(p_name);

	uint32_t idx = hash & STRING_TABLE_MASK;

	_data = _table[idx];

	while (_data) {
		// compare hash first
		if (_data->hash == hash && _data->get_name() == p_name) {
			break;
		}
		_data = _data->next;
	}

	if (_data) {
		if (_data->refcount.ref()) {
			// exists
			if (p_static) {
				_data->static_count.increment();
			}

#ifdef DEBUG_ENABLED
			if (unlikely(debug_stringname)) {
				_data->debug_references++;
			}
#endif

			lock.unlock();
			return;
		}
	}

	_data = memnew(_Data);
	_data->name = p_name;
	_data->refcount.init();
	_data->static_count.set(p_static ? 1 : 0);
	_data->hash = hash;
	_data->idx = idx;
	_data->cname = NULL;
	_data->next = _table[idx];
	_data->prev = NULL;

#ifdef DEBUG_ENABLED
	if (unlikely(debug_stringname)) {
		// Keep in memory, force static.
		_data->refcount.ref();
		_data->static_count.increment();
	}
#endif

	if (_table[idx]) {
		_table[idx]->prev = _data;
	}

	_table[idx] = _data;

	lock.unlock();
}

StringName::StringName(const StaticCString &p_static_string, bool p_static) {
	_data = NULL;

	ERR_FAIL_COND(!configured);

	ERR_FAIL_COND(!p_static_string.ptr || !p_static_string.ptr[0]);

	lock.lock();

	uint32_t hash = String::hash(p_static_string.ptr);

	uint32_t idx = hash & STRING_TABLE_MASK;

	_data = _table[idx];

	while (_data) {
		// compare hash first
		if (_data->hash == hash && _data->get_name() == p_static_string.ptr) {
			break;
		}
		_data = _data->next;
	}

	if (_data) {
		if (_data->refcount.ref()) {
			// exists
			if (p_static) {
				_data->static_count.increment();
			}

#ifdef DEBUG_ENABLED
			if (unlikely(debug_stringname)) {
				_data->debug_references++;
			}
#endif

			lock.unlock();
			return;
		}
	}

	_data = memnew(_Data);

	_data->refcount.init();
	_data->static_count.set(p_static ? 1 : 0);
	_data->hash = hash;
	_data->idx = idx;
	_data->cname = p_static_string.ptr;
	_data->next = _table[idx];
	_data->prev = NULL;

#ifdef DEBUG_ENABLED
	if (unlikely(debug_stringname)) {
		// Keep in memory, force static.
		_data->refcount.ref();
		_data->static_count.increment();
	}
#endif

	if (_table[idx]) {
		_table[idx]->prev = _data;
	}

	_table[idx] = _data;

	lock.unlock();
}

StringName::StringName(const String &p_name, bool p_static) {
	_data = nullptr;

	ERR_FAIL_COND(!configured);

	if (p_name.empty()) {
		return;
	}

	lock.lock();

	uint32_t hash = p_name.hash();

	uint32_t idx = hash & STRING_TABLE_MASK;

	_data = _table[idx];

	while (_data) {
		if (_data->hash == hash && _data->get_name() == p_name) {
			break;
		}

		_data = _data->next;
	}

	if (_data) {
		if (_data->refcount.ref()) {
			// exists
			if (p_static) {
				_data->static_count.increment();
			}

#ifdef DEBUG_ENABLED
			if (unlikely(debug_stringname)) {
				_data->debug_references++;
			}
#endif

			lock.unlock();
			return;
		}
	}

	_data = memnew(_Data);
	_data->name = p_name;
	_data->refcount.init();
	_data->static_count.set(p_static ? 1 : 0);
	_data->hash = hash;
	_data->idx = idx;
	_data->cname = NULL;
	_data->next = _table[idx];
	_data->prev = NULL;

#ifdef DEBUG_ENABLED
	if (unlikely(debug_stringname)) {
		// Keep in memory, force static.
		_data->refcount.ref();
		_data->static_count.increment();
	}
#endif

	if (_table[idx]) {
		_table[idx]->prev = _data;
	}

	_table[idx] = _data;

	lock.unlock();
}

StringName StringName::search(const char *p_name) {
	ERR_FAIL_COND_V(!configured, StringName());

	ERR_FAIL_COND_V(!p_name, StringName());
	if (!p_name[0]) {
		return StringName();
	}

	lock.lock();

	uint32_t hash = String::hash(p_name);

	uint32_t idx = hash & STRING_TABLE_MASK;

	_Data *_data = _table[idx];

	while (_data) {
		// compare hash first
		if (_data->hash == hash && _data->get_name() == p_name) {
			break;
		}
		_data = _data->next;
	}

	if (_data && _data->refcount.ref()) {
#ifdef DEBUG_ENABLED
		if (unlikely(debug_stringname)) {
			_data->debug_references++;
		}
#endif

		lock.unlock();

		return StringName(_data);
	}

	lock.unlock();
	return StringName(); //does not exist
}

StringName StringName::search(const CharType *p_name) {
	ERR_FAIL_COND_V(!configured, StringName());

	ERR_FAIL_COND_V(!p_name, StringName());
	if (!p_name[0]) {
		return StringName();
	}

	lock.lock();

	uint32_t hash = String::hash(p_name);

	uint32_t idx = hash & STRING_TABLE_MASK;

	_Data *_data = _table[idx];

	while (_data) {
		// compare hash first
		if (_data->hash == hash && _data->get_name() == p_name) {
			break;
		}

		_data = _data->next;
	}

	if (_data && _data->refcount.ref()) {
		lock.unlock();
		return StringName(_data);
	}

	lock.unlock();
	return StringName(); //does not exist
}
StringName StringName::search(const String &p_name) {
	ERR_FAIL_COND_V(p_name == "", StringName());

	lock.lock();

	uint32_t hash = p_name.hash();

	uint32_t idx = hash & STRING_TABLE_MASK;

	_Data *_data = _table[idx];

	while (_data) {
		// compare hash first
		if (_data->hash == hash && p_name == _data->get_name()) {
			break;
		}

		_data = _data->next;
	}

	if (_data && _data->refcount.ref()) {
#ifdef DEBUG_ENABLED
		if (unlikely(debug_stringname)) {
			_data->debug_references++;
		}
#endif
		lock.unlock();
		return StringName(_data);
	}

	lock.unlock();
	return StringName(); //does not exist
}

StringName::StringName() {
	_data = nullptr;
}

/*
bool operator==(const String &p_name, const StringName &p_string_name) {
	return p_name == p_string_name.operator String();
}
bool operator!=(const String &p_name, const StringName &p_string_name) {
	return p_name != p_string_name.operator String();
}

bool operator==(const char *p_name, const StringName &p_string_name) {
	return p_name == p_string_name.operator String();
}
bool operator!=(const char *p_name, const StringName &p_string_name) {
	return p_name != p_string_name.operator String();
}
*/
#line 0

#line 1 "sfw/core/aabb.cpp"
/*************************************************************************/
/*  aabb.cpp                                                             */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

real_t AABB::get_volume() const {
	return size.x * size.y * size.z;
}

bool AABB::operator==(const AABB &p_rval) const {
	return ((position == p_rval.position) && (size == p_rval.size));
}
bool AABB::operator!=(const AABB &p_rval) const {
	return ((position != p_rval.position) || (size != p_rval.size));
}

bool AABB::create_from_points(const Vector<Vector3> &p_points) {
	if (!p_points.size()) {
		return false;
	}

	Vector3 begin = p_points[0];
	Vector3 end = begin;

	for (int n = 1; n < p_points.size(); n++) {
		const Vector3 &pt = p_points[n];

		if (pt.x < begin.x) {
			begin.x = pt.x;
		}
		if (pt.y < begin.y) {
			begin.y = pt.y;
		}
		if (pt.z < begin.z) {
			begin.z = pt.z;
		}

		if (pt.x > end.x) {
			end.x = pt.x;
		}
		if (pt.y > end.y) {
			end.y = pt.y;
		}
		if (pt.z > end.z) {
			end.z = pt.z;
		}
	}

	position = begin;
	size = end - begin;

	return true;
}

void AABB::merge_with(const AABB &p_aabb) {
	Vector3 beg_1, beg_2;
	Vector3 end_1, end_2;
	Vector3 min, max;

	beg_1 = position;
	beg_2 = p_aabb.position;
	end_1 = Vector3(size.x, size.y, size.z) + beg_1;
	end_2 = Vector3(p_aabb.size.x, p_aabb.size.y, p_aabb.size.z) + beg_2;

	min.x = (beg_1.x < beg_2.x) ? beg_1.x : beg_2.x;
	min.y = (beg_1.y < beg_2.y) ? beg_1.y : beg_2.y;
	min.z = (beg_1.z < beg_2.z) ? beg_1.z : beg_2.z;

	max.x = (end_1.x > end_2.x) ? end_1.x : end_2.x;
	max.y = (end_1.y > end_2.y) ? end_1.y : end_2.y;
	max.z = (end_1.z > end_2.z) ? end_1.z : end_2.z;

	position = min;
	size = max - min;
}

bool AABB::is_equal_approx(const AABB &p_aabb) const {
	return position.is_equal_approx(p_aabb.position) && size.is_equal_approx(p_aabb.size);
}

AABB AABB::intersection(const AABB &p_aabb) const {
	Vector3 src_min = position;
	Vector3 src_max = position + size;
	Vector3 dst_min = p_aabb.position;
	Vector3 dst_max = p_aabb.position + p_aabb.size;

	Vector3 min, max;

	if (src_min.x > dst_max.x || src_max.x < dst_min.x) {
		return AABB();
	} else {
		min.x = (src_min.x > dst_min.x) ? src_min.x : dst_min.x;
		max.x = (src_max.x < dst_max.x) ? src_max.x : dst_max.x;
	}

	if (src_min.y > dst_max.y || src_max.y < dst_min.y) {
		return AABB();
	} else {
		min.y = (src_min.y > dst_min.y) ? src_min.y : dst_min.y;
		max.y = (src_max.y < dst_max.y) ? src_max.y : dst_max.y;
	}

	if (src_min.z > dst_max.z || src_max.z < dst_min.z) {
		return AABB();
	} else {
		min.z = (src_min.z > dst_min.z) ? src_min.z : dst_min.z;
		max.z = (src_max.z < dst_max.z) ? src_max.z : dst_max.z;
	}

	return AABB(min, max - min);
}

bool AABB::intersects_ray(const Vector3 &p_from, const Vector3 &p_dir, Vector3 *r_clip, Vector3 *r_normal) const {
	Vector3 c1, c2;
	Vector3 end = position + size;
	//<windows.h> has a far and near macro defined, and we can't undo it here in the amalgamation
	real_t aabb_near = -1e20;
	real_t aabb_far = 1e20;
	int axis = 0;

	for (int i = 0; i < 3; i++) {
		if (p_dir[i] == 0) {
			if ((p_from[i] < position[i]) || (p_from[i] > end[i])) {
				return false;
			}
		} else { // ray not parallel to planes in this direction
			c1[i] = (position[i] - p_from[i]) / p_dir[i];
			c2[i] = (end[i] - p_from[i]) / p_dir[i];

			if (c1[i] > c2[i]) {
				SWAP(c1, c2);
			}
			if (c1[i] > aabb_near) {
				aabb_near = c1[i];
				axis = i;
			}
			if (c2[i] < aabb_far) {
				aabb_far = c2[i];
			}
			if ((aabb_near > aabb_far) || (aabb_far < 0)) {
				return false;
			}
		}
	}

	if (r_clip) {
		*r_clip = c1;
	}
	if (r_normal) {
		*r_normal = Vector3();
		(*r_normal)[axis] = p_dir[axis] ? -1 : 1;
	}

	return true;
}

bool AABB::intersects_segment(const Vector3 &p_from, const Vector3 &p_to, Vector3 *r_clip, Vector3 *r_normal) const {
	real_t min = 0, max = 1;
	int axis = 0;
	real_t sign = 0;

	for (int i = 0; i < 3; i++) {
		real_t seg_from = p_from[i];
		real_t seg_to = p_to[i];
		real_t box_begin = position[i];
		real_t box_end = box_begin + size[i];
		real_t cmin, cmax;
		real_t csign;

		if (seg_from < seg_to) {
			if (seg_from > box_end || seg_to < box_begin) {
				return false;
			}
			real_t length = seg_to - seg_from;
			cmin = (seg_from < box_begin) ? ((box_begin - seg_from) / length) : 0;
			cmax = (seg_to > box_end) ? ((box_end - seg_from) / length) : 1;
			csign = -1.0;

		} else {
			if (seg_to > box_end || seg_from < box_begin) {
				return false;
			}
			real_t length = seg_to - seg_from;
			cmin = (seg_from > box_end) ? (box_end - seg_from) / length : 0;
			cmax = (seg_to < box_begin) ? (box_begin - seg_from) / length : 1;
			csign = 1.0;
		}

		if (cmin > min) {
			min = cmin;
			axis = i;
			sign = csign;
		}
		if (cmax < max) {
			max = cmax;
		}
		if (max < min) {
			return false;
		}
	}

	Vector3 rel = p_to - p_from;

	if (r_normal) {
		Vector3 normal;
		normal[axis] = sign;
		*r_normal = normal;
	}

	if (r_clip) {
		*r_clip = p_from + rel * min;
	}

	return true;
}

bool AABB::intersects_plane(const Plane &p_plane) const {
	Vector3 points[8] = {
		Vector3(position.x, position.y, position.z),
		Vector3(position.x, position.y, position.z + size.z),
		Vector3(position.x, position.y + size.y, position.z),
		Vector3(position.x, position.y + size.y, position.z + size.z),
		Vector3(position.x + size.x, position.y, position.z),
		Vector3(position.x + size.x, position.y, position.z + size.z),
		Vector3(position.x + size.x, position.y + size.y, position.z),
		Vector3(position.x + size.x, position.y + size.y, position.z + size.z),
	};

	bool over = false;
	bool under = false;

	for (int i = 0; i < 8; i++) {
		if (p_plane.distance_to(points[i]) > 0) {
			over = true;
		} else {
			under = true;
		}
	}

	return under && over;
}

Vector3 AABB::get_longest_axis() const {
	Vector3 axis(1, 0, 0);
	real_t max_size = size.x;

	if (size.y > max_size) {
		axis = Vector3(0, 1, 0);
		max_size = size.y;
	}

	if (size.z > max_size) {
		axis = Vector3(0, 0, 1);
	}

	return axis;
}
int AABB::get_longest_axis_index() const {
	int axis = 0;
	real_t max_size = size.x;

	if (size.y > max_size) {
		axis = 1;
		max_size = size.y;
	}

	if (size.z > max_size) {
		axis = 2;
	}

	return axis;
}

Vector3 AABB::get_shortest_axis() const {
	Vector3 axis(1, 0, 0);
	real_t max_size = size.x;

	if (size.y < max_size) {
		axis = Vector3(0, 1, 0);
		max_size = size.y;
	}

	if (size.z < max_size) {
		axis = Vector3(0, 0, 1);
	}

	return axis;
}
int AABB::get_shortest_axis_index() const {
	int axis = 0;
	real_t max_size = size.x;

	if (size.y < max_size) {
		axis = 1;
		max_size = size.y;
	}

	if (size.z < max_size) {
		axis = 2;
	}

	return axis;
}

AABB AABB::merge(const AABB &p_with) const {
	AABB aabb = *this;
	aabb.merge_with(p_with);
	return aabb;
}
AABB AABB::expand(const Vector3 &p_vector) const {
	AABB aabb = *this;
	aabb.expand_to(p_vector);
	return aabb;
}
AABB AABB::grow(real_t p_by) const {
	AABB aabb = *this;
	aabb.grow_by(p_by);
	return aabb;
}

void AABB::get_edge(int p_edge, Vector3 &r_from, Vector3 &r_to) const {
	ERR_FAIL_INDEX(p_edge, 12);
	switch (p_edge) {
		case 0: {
			r_from = Vector3(position.x + size.x, position.y, position.z);
			r_to = Vector3(position.x, position.y, position.z);
		} break;
		case 1: {
			r_from = Vector3(position.x + size.x, position.y, position.z + size.z);
			r_to = Vector3(position.x + size.x, position.y, position.z);
		} break;
		case 2: {
			r_from = Vector3(position.x, position.y, position.z + size.z);
			r_to = Vector3(position.x + size.x, position.y, position.z + size.z);

		} break;
		case 3: {
			r_from = Vector3(position.x, position.y, position.z);
			r_to = Vector3(position.x, position.y, position.z + size.z);

		} break;
		case 4: {
			r_from = Vector3(position.x, position.y + size.y, position.z);
			r_to = Vector3(position.x + size.x, position.y + size.y, position.z);
		} break;
		case 5: {
			r_from = Vector3(position.x + size.x, position.y + size.y, position.z);
			r_to = Vector3(position.x + size.x, position.y + size.y, position.z + size.z);
		} break;
		case 6: {
			r_from = Vector3(position.x + size.x, position.y + size.y, position.z + size.z);
			r_to = Vector3(position.x, position.y + size.y, position.z + size.z);

		} break;
		case 7: {
			r_from = Vector3(position.x, position.y + size.y, position.z + size.z);
			r_to = Vector3(position.x, position.y + size.y, position.z);

		} break;
		case 8: {
			r_from = Vector3(position.x, position.y, position.z + size.z);
			r_to = Vector3(position.x, position.y + size.y, position.z + size.z);

		} break;
		case 9: {
			r_from = Vector3(position.x, position.y, position.z);
			r_to = Vector3(position.x, position.y + size.y, position.z);

		} break;
		case 10: {
			r_from = Vector3(position.x + size.x, position.y, position.z);
			r_to = Vector3(position.x + size.x, position.y + size.y, position.z);

		} break;
		case 11: {
			r_from = Vector3(position.x + size.x, position.y, position.z + size.z);
			r_to = Vector3(position.x + size.x, position.y + size.y, position.z + size.z);

		} break;
	}
}

/*
Variant AABB::intersects_segmentv(const Vector3 &p_from, const Vector3 &p_to) const {
	Vector3 inters;
	if (intersects_segment(p_from, p_to, &inters)) {
		return inters;
	}
	return Variant();
}

Variant AABB::intersects_rayv(const Vector3 &p_from, const Vector3 &p_dir) const {
	Vector3 inters;
	if (intersects_ray(p_from, p_dir, &inters)) {
		return inters;
	}
	return Variant();
}
*/

AABB::operator String() const {
	return "[P: " + position.operator String() + ", S: " + size + "]";
}
#line 0

#line 1 "sfw/core/vector3i.cpp"
/*************************************************************************/
/*  vector3i.cpp                                                         */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

void Vector3i::set_axis(const int p_axis, const int32_t p_value) {
	ERR_FAIL_INDEX(p_axis, 3);
	coord[p_axis] = p_value;
}

int32_t Vector3i::get_axis(const int p_axis) const {
	ERR_FAIL_INDEX_V(p_axis, 3, 0);
	return operator[](p_axis);
}

Vector3i::Axis Vector3i::min_axis() const {
	return x < y ? (x < z ? Vector3i::AXIS_X : Vector3i::AXIS_Z) : (y < z ? Vector3i::AXIS_Y : Vector3i::AXIS_Z);
}

Vector3i::Axis Vector3i::max_axis() const {
	return x < y ? (y < z ? Vector3i::AXIS_Z : Vector3i::AXIS_Y) : (x < z ? Vector3i::AXIS_Z : Vector3i::AXIS_X);
}

Vector3i Vector3i::clamp(const Vector3i &p_min, const Vector3i &p_max) const {
	return Vector3i(
			CLAMP(x, p_min.x, p_max.x),
			CLAMP(y, p_min.y, p_max.y),
			CLAMP(z, p_min.z, p_max.z));
}

Vector3 Vector3i::to_vector3() const {
	return Vector3(x, y, z);
}

Vector3i::operator String() const {
	return "(" + itos(x) + ", " + itos(y) + ", " + itos(z) + ")";
}

Vector3i::operator Vector3() const {
	return Vector3(x, y, z);
}
#line 0

#line 1 "sfw/core/transform_2d.cpp"
/*************************************************************************/
/*  transform_2d.cpp                                                     */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

void Transform2D::invert() {
	// FIXME: this function assumes the basis is a rotation matrix, with no scaling.
	// Transform2D::affine_inverse can handle matrices with scaling, so GDScript should eventually use that.
	SWAP(columns[0][1], columns[1][0]);
	columns[2] = basis_xform(-columns[2]);
}

Transform2D Transform2D::inverse() const {
	Transform2D inv = *this;
	inv.invert();
	return inv;
}

void Transform2D::affine_invert() {
	real_t det = basis_determinant();
#ifdef MATH_CHECKS
	ERR_FAIL_COND(det == 0);
#endif
	real_t idet = 1 / det;

	SWAP(columns[0][0], columns[1][1]);
	columns[0] *= Vector2(idet, -idet);
	columns[1] *= Vector2(-idet, idet);

	columns[2] = basis_xform(-columns[2]);
}

Transform2D Transform2D::affine_inverse() const {
	Transform2D inv = *this;
	inv.affine_invert();
	return inv;
}

void Transform2D::rotate(real_t p_phi) {
	*this = Transform2D(p_phi, Vector2()) * (*this);
}

real_t Transform2D::get_rotation() const {
	return Math::atan2(columns[0].y, columns[0].x);
}

void Transform2D::set_rotation(real_t p_rot) {
	Size2 scale = get_scale();
	real_t cr = Math::cos(p_rot);
	real_t sr = Math::sin(p_rot);
	columns[0][0] = cr;
	columns[0][1] = sr;
	columns[1][0] = -sr;
	columns[1][1] = cr;
	set_scale(scale);
}

real_t Transform2D::get_skew() const {
	real_t det = basis_determinant();
	return Math::acos(columns[0].normalized().dot(SGN(det) * columns[1].normalized())) - (real_t)Math_PI * 0.5f;
}

void Transform2D::set_skew(const real_t p_angle) {
	real_t det = basis_determinant();
	columns[1] = SGN(det) * columns[0].rotated(((real_t)Math_PI * 0.5f + p_angle)).normalized() * columns[1].length();
}

Transform2D::Transform2D(real_t p_rot, const Vector2 &p_pos) {
	real_t cr = Math::cos(p_rot);
	real_t sr = Math::sin(p_rot);
	columns[0][0] = cr;
	columns[0][1] = sr;
	columns[1][0] = -sr;
	columns[1][1] = cr;
	columns[2] = p_pos;
}

Transform2D::Transform2D(const real_t p_rot, const Size2 &p_scale, const real_t p_skew, const Vector2 &p_pos) {
	columns[0][0] = Math::cos(p_rot) * p_scale.x;
	columns[1][1] = Math::cos(p_rot + p_skew) * p_scale.y;
	columns[1][0] = -Math::sin(p_rot + p_skew) * p_scale.y;
	columns[0][1] = Math::sin(p_rot) * p_scale.x;
	columns[2] = p_pos;
}

Size2 Transform2D::get_scale() const {
	real_t det_sign = SGN(basis_determinant());
	return Size2(columns[0].length(), det_sign * columns[1].length());
}

void Transform2D::set_scale(const Size2 &p_scale) {
	columns[0].normalize();
	columns[1].normalize();
	columns[0] *= p_scale.x;
	columns[1] *= p_scale.y;
}

void Transform2D::scale(const Size2 &p_scale) {
	scale_basis(p_scale);
	columns[2] *= p_scale;
}
void Transform2D::scale_basis(const Size2 &p_scale) {
	columns[0][0] *= p_scale.x;
	columns[0][1] *= p_scale.y;
	columns[1][0] *= p_scale.x;
	columns[1][1] *= p_scale.y;
}

void Transform2D::translate(real_t p_tx, real_t p_ty) {
	translate(Vector2(p_tx, p_ty));
}
void Transform2D::translate(const Vector2 &p_offset) {
	columns[2] += p_offset;
}

void Transform2D::translate_local(real_t p_tx, real_t p_ty) {
	translate_local(Vector2(p_tx, p_ty));
}
void Transform2D::translate_local(const Vector2 &p_translation) {
	columns[2] += basis_xform(p_translation);
}

void Transform2D::translater(real_t p_tx, real_t p_ty) {
	translate(Vector2(p_tx, p_ty));
}
void Transform2D::translatev(const Vector2 &p_offset) {
	columns[2] += p_offset;
}

void Transform2D::translate_localr(real_t p_tx, real_t p_ty) {
	translate_local(Vector2(p_tx, p_ty));
}
void Transform2D::translate_localv(const Vector2 &p_translation) {
	columns[2] += basis_xform(p_translation);
}

void Transform2D::orthonormalize() {
	// Gram-Schmidt Process

	Vector2 x = columns[0];
	Vector2 y = columns[1];

	x.normalize();
	y = (y - x * (x.dot(y)));
	y.normalize();

	columns[0] = x;
	columns[1] = y;
}

Transform2D Transform2D::orthonormalized() const {
	Transform2D on = *this;
	on.orthonormalize();
	return on;
}

bool Transform2D::is_equal_approx(const Transform2D &p_transform) const {
	return columns[0].is_equal_approx(p_transform.columns[0]) && columns[1].is_equal_approx(p_transform.columns[1]) && columns[2].is_equal_approx(p_transform.columns[2]);
}

Transform2D Transform2D::looking_at(const Vector2 &p_target) const {
	Transform2D return_trans = Transform2D(get_rotation(), get_origin());
	Vector2 target_position = affine_inverse().xform(p_target);
	return_trans.set_rotation(return_trans.get_rotation() + (target_position * get_scale()).angle());
	return return_trans;
}

bool Transform2D::operator==(const Transform2D &p_transform) const {
	for (int i = 0; i < 3; i++) {
		if (columns[i] != p_transform.columns[i]) {
			return false;
		}
	}

	return true;
}

bool Transform2D::operator!=(const Transform2D &p_transform) const {
	for (int i = 0; i < 3; i++) {
		if (columns[i] != p_transform.columns[i]) {
			return true;
		}
	}

	return false;
}

void Transform2D::operator*=(const Transform2D &p_transform) {
	columns[2] = xform(p_transform.columns[2]);

	real_t x0, x1, y0, y1;

	x0 = tdotx(p_transform.columns[0]);
	x1 = tdoty(p_transform.columns[0]);
	y0 = tdotx(p_transform.columns[1]);
	y1 = tdoty(p_transform.columns[1]);

	columns[0][0] = x0;
	columns[0][1] = x1;
	columns[1][0] = y0;
	columns[1][1] = y1;
}

Transform2D Transform2D::operator*(const Transform2D &p_transform) const {
	Transform2D t = *this;
	t *= p_transform;
	return t;
}

void Transform2D::operator*=(const real_t p_val) {
	columns[0] *= p_val;
	columns[1] *= p_val;
	columns[2] *= p_val;
}

Transform2D Transform2D::operator*(const real_t p_val) const {
	Transform2D ret(*this);
	ret *= p_val;
	return ret;
}

Transform2D Transform2D::basis_scaled(const Size2 &p_scale) const {
	Transform2D copy = *this;
	copy.scale_basis(p_scale);
	return copy;
}

Transform2D Transform2D::scaled(const Size2 &p_scale) const {
	// Equivalent to left multiplication
	Transform2D copy = *this;
	copy.scale(p_scale);
	return copy;
}

Transform2D Transform2D::scaled_local(const Size2 &p_scale) const {
	// Equivalent to right multiplication
	return Transform2D(columns[0] * p_scale.x, columns[1] * p_scale.y, columns[2]);
}

Transform2D Transform2D::untranslated() const {
	Transform2D copy = *this;
	copy.columns[2] = Vector2();
	return copy;
}

Transform2D Transform2D::translated(const Vector2 &p_offset) const {
	// Equivalent to left multiplication
	return Transform2D(columns[0], columns[1], columns[2] + p_offset);
}

Transform2D Transform2D::translated_local(const Vector2 &p_offset) const {
	// Equivalent to right multiplication
	return Transform2D(columns[0], columns[1], columns[2] + basis_xform(p_offset));
}

Transform2D Transform2D::rotated(const real_t p_angle) const {
	// Equivalent to left multiplication
	return Transform2D(p_angle, Vector2()) * (*this);
}

Transform2D Transform2D::rotated_local(const real_t p_angle) const {
	// Equivalent to right multiplication
	return (*this) * Transform2D(p_angle, Vector2()); // Could be optimized, because origin transform can be skipped.
}

real_t Transform2D::basis_determinant() const {
	return columns[0].x * columns[1].y - columns[0].y * columns[1].x;
}

Transform2D Transform2D::interpolate_with(const Transform2D &p_transform, real_t p_c) const {
	//extract parameters
	Vector2 p1 = get_origin();
	Vector2 p2 = p_transform.get_origin();

	real_t r1 = get_rotation();
	real_t r2 = p_transform.get_rotation();

	Size2 s1 = get_scale();
	Size2 s2 = p_transform.get_scale();

	//slerp rotation
	Vector2 v1(Math::cos(r1), Math::sin(r1));
	Vector2 v2(Math::cos(r2), Math::sin(r2));

	real_t dot = v1.dot(v2);

	dot = CLAMP(dot, -1, 1);

	Vector2 v;

	if (dot > 0.9995f) {
		v = Vector2::linear_interpolate(v1, v2, p_c).normalized(); //linearly interpolate to avoid numerical precision issues
	} else {
		real_t angle = p_c * Math::acos(dot);
		Vector2 v3 = (v2 - v1 * dot).normalized();
		v = v1 * Math::cos(angle) + v3 * Math::sin(angle);
	}

	//construct matrix
	Transform2D res(Math::atan2(v.y, v.x), Vector2::linear_interpolate(p1, p2, p_c));
	res.scale_basis(Vector2::linear_interpolate(s1, s2, p_c));
	return res;
}

Transform2D::operator String() const {
	return "[X: " + columns[0].operator String() +
			", Y: " + columns[1].operator String() +
			", O: " + columns[2].operator String() + "]";
}
#line 0

#line 1 "sfw/core/projection.cpp"
/*************************************************************************/
/*  projection.cpp                                                       */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

float Projection::determinant() const {
	return matrix[0][3] * matrix[1][2] * matrix[2][1] * matrix[3][0] - matrix[0][2] * matrix[1][3] * matrix[2][1] * matrix[3][0] -
			matrix[0][3] * matrix[1][1] * matrix[2][2] * matrix[3][0] + matrix[0][1] * matrix[1][3] * matrix[2][2] * matrix[3][0] +
			matrix[0][2] * matrix[1][1] * matrix[2][3] * matrix[3][0] - matrix[0][1] * matrix[1][2] * matrix[2][3] * matrix[3][0] -
			matrix[0][3] * matrix[1][2] * matrix[2][0] * matrix[3][1] + matrix[0][2] * matrix[1][3] * matrix[2][0] * matrix[3][1] +
			matrix[0][3] * matrix[1][0] * matrix[2][2] * matrix[3][1] - matrix[0][0] * matrix[1][3] * matrix[2][2] * matrix[3][1] -
			matrix[0][2] * matrix[1][0] * matrix[2][3] * matrix[3][1] + matrix[0][0] * matrix[1][2] * matrix[2][3] * matrix[3][1] +
			matrix[0][3] * matrix[1][1] * matrix[2][0] * matrix[3][2] - matrix[0][1] * matrix[1][3] * matrix[2][0] * matrix[3][2] -
			matrix[0][3] * matrix[1][0] * matrix[2][1] * matrix[3][2] + matrix[0][0] * matrix[1][3] * matrix[2][1] * matrix[3][2] +
			matrix[0][1] * matrix[1][0] * matrix[2][3] * matrix[3][2] - matrix[0][0] * matrix[1][1] * matrix[2][3] * matrix[3][2] -
			matrix[0][2] * matrix[1][1] * matrix[2][0] * matrix[3][3] + matrix[0][1] * matrix[1][2] * matrix[2][0] * matrix[3][3] +
			matrix[0][2] * matrix[1][0] * matrix[2][1] * matrix[3][3] - matrix[0][0] * matrix[1][2] * matrix[2][1] * matrix[3][3] -
			matrix[0][1] * matrix[1][0] * matrix[2][2] * matrix[3][3] + matrix[0][0] * matrix[1][1] * matrix[2][2] * matrix[3][3];
}

void Projection::set_identity() {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			matrix[i][j] = (i == j) ? 1 : 0;
		}
	}
}

void Projection::set_zero() {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			matrix[i][j] = 0;
		}
	}
}

void Projection::adjust_perspective_znear(real_t p_new_znear) {
	real_t zfar = get_z_far();
	real_t znear = p_new_znear;

	real_t deltaZ = zfar - znear;
	matrix[2][2] = -(zfar + znear) / deltaZ;
	matrix[3][2] = -2 * znear * zfar / deltaZ;
}

Projection Projection::create_depth_correction(bool p_flip_y) {
	Projection proj;
	proj.set_depth_correction(p_flip_y);
	return proj;
}

Projection Projection::create_light_atlas_rect(const Rect2 &p_rect) {
	Projection proj;
	proj.set_light_atlas_rect(p_rect);
	return proj;
}

Projection Projection::create_perspective(real_t p_fovy_degrees, real_t p_aspect, real_t p_z_near, real_t p_z_far, bool p_flip_fov) {
	Projection proj;
	proj.set_perspective(p_fovy_degrees, p_aspect, p_z_near, p_z_far, p_flip_fov);
	return proj;
}

Projection Projection::create_perspective_hmd(real_t p_fovy_degrees, real_t p_aspect, real_t p_z_near, real_t p_z_far, bool p_flip_fov, int p_eye, real_t p_intraocular_dist, real_t p_convergence_dist) {
	Projection proj;
	proj.set_perspective(p_fovy_degrees, p_aspect, p_z_near, p_z_far, p_flip_fov, p_eye, p_intraocular_dist, p_convergence_dist);
	return proj;
}

Projection Projection::create_for_hmd(int p_eye, real_t p_aspect, real_t p_intraocular_dist, real_t p_display_width, real_t p_display_to_lens, real_t p_oversample, real_t p_z_near, real_t p_z_far) {
	Projection proj;
	proj.set_for_hmd(p_eye, p_aspect, p_intraocular_dist, p_display_width, p_display_to_lens, p_oversample, p_z_near, p_z_far);
	return proj;
}

Projection Projection::create_orthogonal(real_t p_left, real_t p_right, real_t p_bottom, real_t p_top, real_t p_znear, real_t p_zfar) {
	Projection proj;
	proj.set_orthogonal(p_left, p_right, p_bottom, p_top, p_zfar, p_zfar);
	return proj;
}

Projection Projection::create_orthogonal_aspect(real_t p_size, real_t p_aspect, real_t p_znear, real_t p_zfar, bool p_flip_fov) {
	Projection proj;
	proj.set_orthogonal(p_size, p_aspect, p_znear, p_zfar, p_flip_fov);
	return proj;
}

Projection Projection::create_frustum(real_t p_left, real_t p_right, real_t p_bottom, real_t p_top, real_t p_near, real_t p_far) {
	Projection proj;
	proj.set_frustum(p_left, p_right, p_bottom, p_top, p_near, p_far);
	return proj;
}

Projection Projection::create_frustum_aspect(real_t p_size, real_t p_aspect, Vector2 p_offset, real_t p_near, real_t p_far, bool p_flip_fov) {
	Projection proj;
	proj.set_frustum(p_size, p_aspect, p_offset, p_near, p_far, p_flip_fov);
	return proj;
}

Projection Projection::create_fit_aabb(const AABB &p_aabb) {
	Projection proj;
	proj.scale_translate_to_fit(p_aabb);
	return proj;
}

Projection Projection::perspective_znear_adjusted(real_t p_new_znear) const {
	Projection proj = *this;
	proj.adjust_perspective_znear(p_new_znear);
	return proj;
}

Plane Projection::get_projection_plane(Projection::Planes p_plane) const {
	const real_t *matrix = (const real_t *)this->matrix;

	switch (p_plane) {
		case PLANE_NEAR: {
			Plane new_plane = Plane(matrix[3] + matrix[2],
					matrix[7] + matrix[6],
					matrix[11] + matrix[10],
					matrix[15] + matrix[14]);

			new_plane.normal = -new_plane.normal;
			new_plane.normalize();
			return new_plane;
		} break;
		case PLANE_FAR: {
			Plane new_plane = Plane(matrix[3] - matrix[2],
					matrix[7] - matrix[6],
					matrix[11] - matrix[10],
					matrix[15] - matrix[14]);

			new_plane.normal = -new_plane.normal;
			new_plane.normalize();
			return new_plane;
		} break;
		case PLANE_LEFT: {
			Plane new_plane = Plane(matrix[3] + matrix[0],
					matrix[7] + matrix[4],
					matrix[11] + matrix[8],
					matrix[15] + matrix[12]);

			new_plane.normal = -new_plane.normal;
			new_plane.normalize();
			return new_plane;
		} break;
		case PLANE_TOP: {
			Plane new_plane = Plane(matrix[3] - matrix[1],
					matrix[7] - matrix[5],
					matrix[11] - matrix[9],
					matrix[15] - matrix[13]);

			new_plane.normal = -new_plane.normal;
			new_plane.normalize();
			return new_plane;
		} break;
		case PLANE_RIGHT: {
			Plane new_plane = Plane(matrix[3] - matrix[0],
					matrix[7] - matrix[4],
					matrix[11] - matrix[8],
					matrix[15] - matrix[12]);

			new_plane.normal = -new_plane.normal;
			new_plane.normalize();
			return new_plane;
		} break;
		case PLANE_BOTTOM: {
			Plane new_plane = Plane(matrix[3] + matrix[1],
					matrix[7] + matrix[5],
					matrix[11] + matrix[9],
					matrix[15] + matrix[13]);

			new_plane.normal = -new_plane.normal;
			new_plane.normalize();
			return new_plane;
		} break;
	}

	return Plane();
}

Projection Projection::flipped_y() const {
	Projection proj = *this;
	proj.flip_y();
	return proj;
}

Projection Projection ::jitter_offseted(const Vector2 &p_offset) const {
	Projection proj = *this;
	proj.add_jitter_offset(p_offset);
	return proj;
}

void Projection::set_perspective(real_t p_fovy_degrees, real_t p_aspect, real_t p_z_near, real_t p_z_far, bool p_flip_fov) {
	if (p_flip_fov) {
		p_fovy_degrees = get_fovy(p_fovy_degrees, 1.0 / p_aspect);
	}

	real_t sine, cotangent, deltaZ;
	real_t radians = Math::deg2rad(p_fovy_degrees / 2.0);

	deltaZ = p_z_far - p_z_near;
	sine = Math::sin(radians);

	if ((deltaZ == 0) || (sine == 0) || (p_aspect == 0)) {
		return;
	}
	cotangent = Math::cos(radians) / sine;

	set_identity();

	matrix[0][0] = cotangent / p_aspect;
	matrix[1][1] = cotangent;
	matrix[2][2] = -(p_z_far + p_z_near) / deltaZ;
	matrix[2][3] = -1;
	matrix[3][2] = -2 * p_z_near * p_z_far / deltaZ;
	matrix[3][3] = 0;
}

void Projection::set_perspective(real_t p_fovy_degrees, real_t p_aspect, real_t p_z_near, real_t p_z_far, bool p_flip_fov, int p_eye, real_t p_intraocular_dist, real_t p_convergence_dist) {
	if (p_flip_fov) {
		p_fovy_degrees = get_fovy(p_fovy_degrees, 1.0 / p_aspect);
	}

	real_t left, right, modeltranslation, ymax, xmax, frustumshift;

	ymax = p_z_near * tan(Math::deg2rad(p_fovy_degrees / 2.0));
	xmax = ymax * p_aspect;
	frustumshift = (p_intraocular_dist / 2.0) * p_z_near / p_convergence_dist;

	switch (p_eye) {
		case 1: { // left eye
			left = -xmax + frustumshift;
			right = xmax + frustumshift;
			modeltranslation = p_intraocular_dist / 2.0;
		} break;
		case 2: { // right eye
			left = -xmax - frustumshift;
			right = xmax - frustumshift;
			modeltranslation = -p_intraocular_dist / 2.0;
		} break;
		default: { // mono, should give the same result as set_perspective(p_fovy_degrees,p_aspect,p_z_near,p_z_far,p_flip_fov)
			left = -xmax;
			right = xmax;
			modeltranslation = 0.0;
		} break;
	}

	set_frustum(left, right, -ymax, ymax, p_z_near, p_z_far);

	// translate matrix by (modeltranslation, 0.0, 0.0)
	Projection cm;
	cm.set_identity();
	cm.matrix[3][0] = modeltranslation;
	*this = *this * cm;
}

void Projection::set_for_hmd(int p_eye, real_t p_aspect, real_t p_intraocular_dist, real_t p_display_width, real_t p_display_to_lens, real_t p_oversample, real_t p_z_near, real_t p_z_far) {
	// we first calculate our base frustum on our values without taking our lens magnification into account.
	real_t f1 = (p_intraocular_dist * 0.5) / p_display_to_lens;
	real_t f2 = ((p_display_width - p_intraocular_dist) * 0.5) / p_display_to_lens;
	real_t f3 = (p_display_width / 4.0) / p_display_to_lens;

	// now we apply our oversample factor to increase our FOV. how much we oversample is always a balance we strike between performance and how much
	// we're willing to sacrifice in FOV.
	real_t add = ((f1 + f2) * (p_oversample - 1.0)) / 2.0;
	f1 += add;
	f2 += add;
	f3 *= p_oversample;

	// always apply KEEP_WIDTH aspect ratio
	f3 /= p_aspect;

	switch (p_eye) {
		case 1: { // left eye
			set_frustum(-f2 * p_z_near, f1 * p_z_near, -f3 * p_z_near, f3 * p_z_near, p_z_near, p_z_far);
		} break;
		case 2: { // right eye
			set_frustum(-f1 * p_z_near, f2 * p_z_near, -f3 * p_z_near, f3 * p_z_near, p_z_near, p_z_far);
		} break;
		default: { // mono, does not apply here!
		} break;
	}
}

void Projection::set_orthogonal(real_t p_left, real_t p_right, real_t p_bottom, real_t p_top, real_t p_znear, real_t p_zfar) {
	set_identity();

	matrix[0][0] = 2.0 / (p_right - p_left);
	matrix[3][0] = -((p_right + p_left) / (p_right - p_left));
	matrix[1][1] = 2.0 / (p_top - p_bottom);
	matrix[3][1] = -((p_top + p_bottom) / (p_top - p_bottom));
	matrix[2][2] = -2.0 / (p_zfar - p_znear);
	matrix[3][2] = -((p_zfar + p_znear) / (p_zfar - p_znear));
	matrix[3][3] = 1.0;
}

void Projection::set_orthogonal(real_t p_size, real_t p_aspect, real_t p_znear, real_t p_zfar, bool p_flip_fov) {
	if (!p_flip_fov) {
		p_size *= p_aspect;
	}

	set_orthogonal(-p_size / 2, +p_size / 2, -p_size / p_aspect / 2, +p_size / p_aspect / 2, p_znear, p_zfar);
}

void Projection::set_frustum(real_t p_left, real_t p_right, real_t p_bottom, real_t p_top, real_t p_near, real_t p_far) {
	ERR_FAIL_COND(p_right <= p_left);
	ERR_FAIL_COND(p_top <= p_bottom);
	ERR_FAIL_COND(p_far <= p_near);

	real_t *te = &matrix[0][0];
	real_t x = 2 * p_near / (p_right - p_left);
	real_t y = 2 * p_near / (p_top - p_bottom);

	real_t a = (p_right + p_left) / (p_right - p_left);
	real_t b = (p_top + p_bottom) / (p_top - p_bottom);
	real_t c = -(p_far + p_near) / (p_far - p_near);
	real_t d = -2 * p_far * p_near / (p_far - p_near);

	te[0] = x;
	te[1] = 0;
	te[2] = 0;
	te[3] = 0;
	te[4] = 0;
	te[5] = y;
	te[6] = 0;
	te[7] = 0;
	te[8] = a;
	te[9] = b;
	te[10] = c;
	te[11] = -1;
	te[12] = 0;
	te[13] = 0;
	te[14] = d;
	te[15] = 0;
}

void Projection::set_frustum(real_t p_size, real_t p_aspect, Vector2 p_offset, real_t p_near, real_t p_far, bool p_flip_fov) {
	if (!p_flip_fov) {
		p_size *= p_aspect;
	}

	set_frustum(-p_size / 2 + p_offset.x, +p_size / 2 + p_offset.x, -p_size / p_aspect / 2 + p_offset.y, +p_size / p_aspect / 2 + p_offset.y, p_near, p_far);
}

real_t Projection::get_z_far() const {
	const real_t *matrix = (const real_t *)this->matrix;
	Plane new_plane = Plane(matrix[3] - matrix[2],
			matrix[7] - matrix[6],
			matrix[11] - matrix[10],
			matrix[15] - matrix[14]);

	new_plane.normal = -new_plane.normal;
	new_plane.normalize();

	return new_plane.d;
}

real_t Projection::get_z_near() const {
	const real_t *matrix = (const real_t *)this->matrix;
	Plane new_plane = Plane(matrix[3] + matrix[2],
			matrix[7] + matrix[6],
			matrix[11] + matrix[10],
			-matrix[15] - matrix[14]);

	new_plane.normalize();
	return new_plane.d;
}

Vector2 Projection::get_viewport_half_extents() const {
	const real_t *matrix = (const real_t *)this->matrix;
	///////--- Near Plane ---///////
	Plane near_plane = Plane(matrix[3] + matrix[2],
			matrix[7] + matrix[6],
			matrix[11] + matrix[10],
			-matrix[15] - matrix[14]);
	near_plane.normalize();

	///////--- Right Plane ---///////
	Plane right_plane = Plane(matrix[3] - matrix[0],
			matrix[7] - matrix[4],
			matrix[11] - matrix[8],
			-matrix[15] + matrix[12]);
	right_plane.normalize();

	Plane top_plane = Plane(matrix[3] - matrix[1],
			matrix[7] - matrix[5],
			matrix[11] - matrix[9],
			-matrix[15] + matrix[13]);
	top_plane.normalize();

	Vector3 res;
	near_plane.intersect_3(right_plane, top_plane, &res);

	return Vector2(res.x, res.y);
}

Vector2 Projection::get_far_plane_half_extents() const {
	const real_t *matrix = (const real_t *)this->matrix;
	///////--- Far Plane ---///////
	Plane far_plane = Plane(matrix[3] - matrix[2],
			matrix[7] - matrix[6],
			matrix[11] - matrix[10],
			-matrix[15] + matrix[14]);
	far_plane.normalize();

	///////--- Right Plane ---///////
	Plane right_plane = Plane(matrix[3] - matrix[0],
			matrix[7] - matrix[4],
			matrix[11] - matrix[8],
			-matrix[15] + matrix[12]);
	right_plane.normalize();

	Plane top_plane = Plane(matrix[3] - matrix[1],
			matrix[7] - matrix[5],
			matrix[11] - matrix[9],
			-matrix[15] + matrix[13]);
	top_plane.normalize();

	Vector3 res;
	far_plane.intersect_3(right_plane, top_plane, &res);

	return Vector2(res.x, res.y);
}

bool Projection::get_endpoints(const Transform &p_transform, Vector3 *p_8points) const {
	Vector<Plane> planes = get_projection_planes(Transform());
	const Planes intersections[8][3] = {
		{ PLANE_FAR, PLANE_LEFT, PLANE_TOP },
		{ PLANE_FAR, PLANE_LEFT, PLANE_BOTTOM },
		{ PLANE_FAR, PLANE_RIGHT, PLANE_TOP },
		{ PLANE_FAR, PLANE_RIGHT, PLANE_BOTTOM },
		{ PLANE_NEAR, PLANE_LEFT, PLANE_TOP },
		{ PLANE_NEAR, PLANE_LEFT, PLANE_BOTTOM },
		{ PLANE_NEAR, PLANE_RIGHT, PLANE_TOP },
		{ PLANE_NEAR, PLANE_RIGHT, PLANE_BOTTOM },
	};

	for (int i = 0; i < 8; i++) {
		Vector3 point;
		bool res = planes[intersections[i][0]].intersect_3(planes[intersections[i][1]], planes[intersections[i][2]], &point);
		ERR_FAIL_COND_V(!res, false);
		p_8points[i] = p_transform.xform(point);
	}

	return true;
}

Vector<Plane> Projection::get_projection_planes(const Transform &p_transform) const {
	/** Fast Plane Extraction from combined modelview/projection matrices.
	 * References:
	 * https://web.archive.org/web/20011221205252/https://www.markmorley.com/opengl/frustumculling.html
	 * https://web.archive.org/web/20061020020112/https://www2.ravensoft.com/users/ggribb/plane%20extraction.pdf
	 */

	Vector<Plane> planes;
	planes.resize(6);

	const real_t *matrix = (const real_t *)this->matrix;

	Plane new_plane;

	///////--- Near Plane ---///////
	new_plane = Plane(matrix[3] + matrix[2],
			matrix[7] + matrix[6],
			matrix[11] + matrix[10],
			matrix[15] + matrix[14]);

	new_plane.normal = -new_plane.normal;
	new_plane.normalize();

	planes.write[0] = p_transform.xform(new_plane);

	///////--- Far Plane ---///////
	new_plane = Plane(matrix[3] - matrix[2],
			matrix[7] - matrix[6],
			matrix[11] - matrix[10],
			matrix[15] - matrix[14]);

	new_plane.normal = -new_plane.normal;
	new_plane.normalize();

	planes.write[1] = p_transform.xform(new_plane);

	///////--- Left Plane ---///////
	new_plane = Plane(matrix[3] + matrix[0],
			matrix[7] + matrix[4],
			matrix[11] + matrix[8],
			matrix[15] + matrix[12]);

	new_plane.normal = -new_plane.normal;
	new_plane.normalize();

	planes.write[2] = p_transform.xform(new_plane);

	///////--- Top Plane ---///////
	new_plane = Plane(matrix[3] - matrix[1],
			matrix[7] - matrix[5],
			matrix[11] - matrix[9],
			matrix[15] - matrix[13]);

	new_plane.normal = -new_plane.normal;
	new_plane.normalize();

	planes.write[3] = p_transform.xform(new_plane);

	///////--- Right Plane ---///////
	new_plane = Plane(matrix[3] - matrix[0],
			matrix[7] - matrix[4],
			matrix[11] - matrix[8],
			matrix[15] - matrix[12]);

	new_plane.normal = -new_plane.normal;
	new_plane.normalize();

	planes.write[4] = p_transform.xform(new_plane);

	///////--- Bottom Plane ---///////
	new_plane = Plane(matrix[3] + matrix[1],
			matrix[7] + matrix[5],
			matrix[11] + matrix[9],
			matrix[15] + matrix[13]);

	new_plane.normal = -new_plane.normal;
	new_plane.normalize();

	planes.write[5] = p_transform.xform(new_plane);

	return planes;
}

Projection Projection::inverse() const {
	Projection cm = *this;
	cm.invert();
	return cm;
}

void Projection::invert() {
	int i, j, k;
	int pvt_i[4], pvt_j[4]; /* Locations of pivot matrix */
	real_t pvt_val; /* Value of current pivot element */
	real_t hold; /* Temporary storage */
	real_t determinant = 1.0f;
	for (k = 0; k < 4; k++) {
		/** Locate k'th pivot element **/
		pvt_val = matrix[k][k]; /** Initialize for search **/
		pvt_i[k] = k;
		pvt_j[k] = k;
		for (i = k; i < 4; i++) {
			for (j = k; j < 4; j++) {
				if (Math::abs(matrix[i][j]) > Math::abs(pvt_val)) {
					pvt_i[k] = i;
					pvt_j[k] = j;
					pvt_val = matrix[i][j];
				}
			}
		}

		/** Product of pivots, gives determinant when finished **/
		determinant *= pvt_val;
		if (Math::is_zero_approx(determinant)) {
			return; /** Matrix is singular (zero determinant). **/
		}

		/** "Interchange" rows (with sign change stuff) **/
		i = pvt_i[k];
		if (i != k) { /** If rows are different **/
			for (j = 0; j < 4; j++) {
				hold = -matrix[k][j];
				matrix[k][j] = matrix[i][j];
				matrix[i][j] = hold;
			}
		}

		/** "Interchange" columns **/
		j = pvt_j[k];
		if (j != k) { /** If columns are different **/
			for (i = 0; i < 4; i++) {
				hold = -matrix[i][k];
				matrix[i][k] = matrix[i][j];
				matrix[i][j] = hold;
			}
		}

		/** Divide column by minus pivot value **/
		for (i = 0; i < 4; i++) {
			if (i != k) {
				matrix[i][k] /= (-pvt_val);
			}
		}

		/** Reduce the matrix **/
		for (i = 0; i < 4; i++) {
			hold = matrix[i][k];
			for (j = 0; j < 4; j++) {
				if (i != k && j != k) {
					matrix[i][j] += hold * matrix[k][j];
				}
			}
		}

		/** Divide row by pivot **/
		for (j = 0; j < 4; j++) {
			if (j != k) {
				matrix[k][j] /= pvt_val;
			}
		}

		/** Replace pivot by reciprocal (at last we can touch it). **/
		matrix[k][k] = 1.0 / pvt_val;
	}

	/* That was most of the work, one final pass of row/column interchange */
	/* to finish */
	for (k = 4 - 2; k >= 0; k--) { /* Don't need to work with 1 by 1 corner*/
		i = pvt_j[k]; /* Rows to swap correspond to pivot COLUMN */
		if (i != k) { /* If rows are different */
			for (j = 0; j < 4; j++) {
				hold = matrix[k][j];
				matrix[k][j] = -matrix[i][j];
				matrix[i][j] = hold;
			}
		}

		j = pvt_i[k]; /* Columns to swap correspond to pivot ROW */
		if (j != k) { /* If columns are different */
			for (i = 0; i < 4; i++) {
				hold = matrix[i][k];
				matrix[i][k] = -matrix[i][j];
				matrix[i][j] = hold;
			}
		}
	}
}

void Projection::flip_y() {
	for (int i = 0; i < 4; i++) {
		matrix[1][i] = -matrix[1][i];
	}
}

Projection::Projection() {
	set_identity();
}

Projection Projection::operator*(const Projection &p_matrix) const {
	Projection new_matrix;

	for (int j = 0; j < 4; j++) {
		for (int i = 0; i < 4; i++) {
			real_t ab = 0;
			for (int k = 0; k < 4; k++) {
				ab += matrix[k][i] * p_matrix.matrix[j][k];
			}
			new_matrix.matrix[j][i] = ab;
		}
	}

	return new_matrix;
}

void Projection::set_depth_correction(bool p_flip_y) {
	real_t *m = &matrix[0][0];

	m[0] = 1;
	m[1] = 0.0;
	m[2] = 0.0;
	m[3] = 0.0;
	m[4] = 0.0;
	m[5] = p_flip_y ? -1 : 1;
	m[6] = 0.0;
	m[7] = 0.0;
	m[8] = 0.0;
	m[9] = 0.0;
	m[10] = 0.5;
	m[11] = 0.0;
	m[12] = 0.0;
	m[13] = 0.0;
	m[14] = 0.5;
	m[15] = 1.0;
}

void Projection::set_light_bias() {
	real_t *m = &matrix[0][0];

	m[0] = 0.5;
	m[1] = 0.0;
	m[2] = 0.0;
	m[3] = 0.0;
	m[4] = 0.0;
	m[5] = 0.5;
	m[6] = 0.0;
	m[7] = 0.0;
	m[8] = 0.0;
	m[9] = 0.0;
	m[10] = 0.5;
	m[11] = 0.0;
	m[12] = 0.5;
	m[13] = 0.5;
	m[14] = 0.5;
	m[15] = 1.0;
}

void Projection::set_light_atlas_rect(const Rect2 &p_rect) {
	real_t *m = &matrix[0][0];

	m[0] = p_rect.size.width;
	m[1] = 0.0;
	m[2] = 0.0;
	m[3] = 0.0;
	m[4] = 0.0;
	m[5] = p_rect.size.height;
	m[6] = 0.0;
	m[7] = 0.0;
	m[8] = 0.0;
	m[9] = 0.0;
	m[10] = 1.0;
	m[11] = 0.0;
	m[12] = p_rect.position.x;
	m[13] = p_rect.position.y;
	m[14] = 0.0;
	m[15] = 1.0;
}

Vector4 Projection::xform(const Vector4 &p_vec4) const {
	return Vector4(
			matrix[0][0] * p_vec4.x + matrix[1][0] * p_vec4.y + matrix[2][0] * p_vec4.z + matrix[3][0] * p_vec4.w,
			matrix[0][1] * p_vec4.x + matrix[1][1] * p_vec4.y + matrix[2][1] * p_vec4.z + matrix[3][1] * p_vec4.w,
			matrix[0][2] * p_vec4.x + matrix[1][2] * p_vec4.y + matrix[2][2] * p_vec4.z + matrix[3][2] * p_vec4.w,
			matrix[0][3] * p_vec4.x + matrix[1][3] * p_vec4.y + matrix[2][3] * p_vec4.z + matrix[3][3] * p_vec4.w);
}

Vector4 Projection::xform_inv(const Vector4 &p_vec4) const {
	return Vector4(
			matrix[0][0] * p_vec4.x + matrix[0][1] * p_vec4.y + matrix[0][2] * p_vec4.z + matrix[0][3] * p_vec4.w,
			matrix[1][0] * p_vec4.x + matrix[1][1] * p_vec4.y + matrix[1][2] * p_vec4.z + matrix[1][3] * p_vec4.w,
			matrix[2][0] * p_vec4.x + matrix[2][1] * p_vec4.y + matrix[2][2] * p_vec4.z + matrix[2][3] * p_vec4.w,
			matrix[3][0] * p_vec4.x + matrix[3][1] * p_vec4.y + matrix[3][2] * p_vec4.z + matrix[3][3] * p_vec4.w);
}

Plane Projection::xform(const Plane &p_vec4) const {
	Plane ret;

	ret.normal.x = matrix[0][0] * p_vec4.normal.x + matrix[1][0] * p_vec4.normal.y + matrix[2][0] * p_vec4.normal.z + matrix[3][0] * p_vec4.d;
	ret.normal.y = matrix[0][1] * p_vec4.normal.x + matrix[1][1] * p_vec4.normal.y + matrix[2][1] * p_vec4.normal.z + matrix[3][1] * p_vec4.d;
	ret.normal.z = matrix[0][2] * p_vec4.normal.x + matrix[1][2] * p_vec4.normal.y + matrix[2][2] * p_vec4.normal.z + matrix[3][2] * p_vec4.d;
	ret.d = matrix[0][3] * p_vec4.normal.x + matrix[1][3] * p_vec4.normal.y + matrix[2][3] * p_vec4.normal.z + matrix[3][3] * p_vec4.d;
	return ret;
}

Projection::operator String() const {
	return "[ X: " + matrix[0].operator String() +
			", Y: " + matrix[1].operator String() +
			", Z: " + matrix[2].operator String() +
			", W: " + matrix[3].operator String() + " ]";
}

real_t Projection::get_aspect() const {
	Vector2 vp_he = get_viewport_half_extents();
	return vp_he.x / vp_he.y;
}

int Projection::get_pixels_per_meter(int p_for_pixel_width) const {
	Vector3 result = xform(Vector3(1, 0, -1));

	return int((result.x * 0.5 + 0.5) * p_for_pixel_width);
}

bool Projection::is_orthogonal() const {
	return matrix[3][3] == 1.0;
}

real_t Projection::get_fov() const {
	const real_t *matrix = (const real_t *)this->matrix;

	Plane right_plane = Plane(matrix[3] - matrix[0],
			matrix[7] - matrix[4],
			matrix[11] - matrix[8],
			-matrix[15] + matrix[12]);
	right_plane.normalize();

	if ((matrix[8] == 0) && (matrix[9] == 0)) {
		return Math::rad2deg(Math::acos(Math::abs(right_plane.normal.x))) * 2.0;
	} else {
		// our frustum is asymmetrical need to calculate the left planes angle separately..
		Plane left_plane = Plane(matrix[3] + matrix[0],
				matrix[7] + matrix[4],
				matrix[11] + matrix[8],
				matrix[15] + matrix[12]);
		left_plane.normalize();

		return Math::rad2deg(Math::acos(Math::abs(left_plane.normal.x))) + Math::rad2deg(Math::acos(Math::abs(right_plane.normal.x)));
	}
}

float Projection::get_lod_multiplier() const {
	if (is_orthogonal()) {
		return get_viewport_half_extents().x;
	} else {
		float zn = get_z_near();
		float width = get_viewport_half_extents().x * 2.0;
		return 1.0 / (zn / width);
	}

	//usage is lod_size / (lod_distance * multiplier) < threshold
}
void Projection::make_scale(const Vector3 &p_scale) {
	set_identity();
	matrix[0][0] = p_scale.x;
	matrix[1][1] = p_scale.y;
	matrix[2][2] = p_scale.z;
}

void Projection::scale_translate_to_fit(const AABB &p_aabb) {
	Vector3 min = p_aabb.position;
	Vector3 max = p_aabb.position + p_aabb.size;

	matrix[0][0] = 2 / (max.x - min.x);
	matrix[1][0] = 0;
	matrix[2][0] = 0;
	matrix[3][0] = -(max.x + min.x) / (max.x - min.x);

	matrix[0][1] = 0;
	matrix[1][1] = 2 / (max.y - min.y);
	matrix[2][1] = 0;
	matrix[3][1] = -(max.y + min.y) / (max.y - min.y);

	matrix[0][2] = 0;
	matrix[1][2] = 0;
	matrix[2][2] = 2 / (max.z - min.z);
	matrix[3][2] = -(max.z + min.z) / (max.z - min.z);

	matrix[0][3] = 0;
	matrix[1][3] = 0;
	matrix[2][3] = 0;
	matrix[3][3] = 1;
}

void Projection::add_jitter_offset(const Vector2 &p_offset) {
	matrix[3][0] += p_offset.x;
	matrix[3][1] += p_offset.y;
}

Projection::operator Transform() const {
	Transform tr;
	const real_t *m = &matrix[0][0];

	tr.basis.rows[0][0] = m[0];
	tr.basis.rows[1][0] = m[1];
	tr.basis.rows[2][0] = m[2];

	tr.basis.rows[0][1] = m[4];
	tr.basis.rows[1][1] = m[5];
	tr.basis.rows[2][1] = m[6];

	tr.basis.rows[0][2] = m[8];
	tr.basis.rows[1][2] = m[9];
	tr.basis.rows[2][2] = m[10];

	tr.origin.x = m[12];
	tr.origin.y = m[13];
	tr.origin.z = m[14];

	return tr;
}

void Projection::set_frustum2(real_t p_size, real_t p_aspect, Vector2 p_offset, real_t p_near, real_t p_far, bool p_flip_fov) {
	set_frustum(p_size, p_aspect, p_offset, p_near, p_far, p_flip_fov);
}

Projection::Projection(const Vector4 &p_x, const Vector4 &p_y, const Vector4 &p_z, const Vector4 &p_w) {
	matrix[0] = p_x;
	matrix[1] = p_y;
	matrix[2] = p_z;
	matrix[3] = p_w;
}
Projection::Projection(const Transform &p_transform) {
	const Transform &tr = p_transform;
	real_t *m = &matrix[0][0];

	m[0] = tr.basis.rows[0][0];
	m[1] = tr.basis.rows[1][0];
	m[2] = tr.basis.rows[2][0];
	m[3] = 0.0;
	m[4] = tr.basis.rows[0][1];
	m[5] = tr.basis.rows[1][1];
	m[6] = tr.basis.rows[2][1];
	m[7] = 0.0;
	m[8] = tr.basis.rows[0][2];
	m[9] = tr.basis.rows[1][2];
	m[10] = tr.basis.rows[2][2];
	m[11] = 0.0;
	m[12] = tr.origin.x;
	m[13] = tr.origin.y;
	m[14] = tr.origin.z;
	m[15] = 1.0;
}

Projection::~Projection() {
}
#line 0

#line 1 "sfw/core/vector3.cpp"
/*************************************************************************/
/*  vector3.cpp                                                          */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

void Vector3::rotate(const Vector3 &p_axis, real_t p_phi) {
	*this = Basis(p_axis, p_phi).xform(*this);
}

Vector3 Vector3::rotated(const Vector3 &p_axis, real_t p_phi) const {
	Vector3 r = *this;
	r.rotate(p_axis, p_phi);
	return r;
}

void Vector3::set_axis(int p_axis, real_t p_value) {
	ERR_FAIL_INDEX(p_axis, 3);
	coord[p_axis] = p_value;
}
real_t Vector3::get_axis(int p_axis) const {
	ERR_FAIL_INDEX_V(p_axis, 3, 0);
	return operator[](p_axis);
}

void Vector3::snap(const Vector3 &p_val) {
	x = Math::stepify(x, p_val.x);
	y = Math::stepify(y, p_val.y);
	z = Math::stepify(z, p_val.z);
}
Vector3 Vector3::snapped(const Vector3 &p_val) const {
	Vector3 v = *this;
	v.snap(p_val);
	return v;
}

Vector3 Vector3::limit_length(const real_t p_len) const {
	const real_t l = length();
	Vector3 v = *this;
	if (l > 0 && p_len < l) {
		v /= l;
		v *= p_len;
	}

	return v;
}

Vector3 Vector3::move_toward(const Vector3 &p_to, const real_t p_delta) const {
	Vector3 v = *this;
	Vector3 vd = p_to - v;
	real_t len = vd.length();
	return len <= p_delta || len < (real_t)CMP_EPSILON ? p_to : v + vd / len * p_delta;
}

Basis Vector3::outer(const Vector3 &p_b) const {
	Vector3 row0(x * p_b.x, x * p_b.y, x * p_b.z);
	Vector3 row1(y * p_b.x, y * p_b.y, y * p_b.z);
	Vector3 row2(z * p_b.x, z * p_b.y, z * p_b.z);

	return Basis(row0, row1, row2);
}

Basis Vector3::to_diagonal_matrix() const {
	return Basis(x, 0, 0,
			0, y, 0,
			0, 0, z);
}

Vector3 Vector3::clamp(const Vector3 &p_min, const Vector3 &p_max) const {
	return Vector3(
			CLAMP(x, p_min.x, p_max.x),
			CLAMP(y, p_min.y, p_max.y),
			CLAMP(z, p_min.z, p_max.z));
}

bool Vector3::is_equal_approx(const Vector3 &p_v) const {
	return Math::is_equal_approx(x, p_v.x) && Math::is_equal_approx(y, p_v.y) && Math::is_equal_approx(z, p_v.z);
}

Vector3::operator String() const {
	return "(" + String::num_real(x) + ", " + String::num_real(y) + ", " + String::num_real(z) + ")";
}
#line 0

#line 1 "sfw/core/pcg.cpp"
// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)

uint32_t pcg32_random_r(pcg32_random_t *rng) {
	uint64_t oldstate = rng->state;
	// Advance internal state
	rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
	// Calculate output function (XSH RR), uses old state for max ILP
	uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
	uint32_t rot = oldstate >> 59u;
	return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

// Source from http://www.pcg-random.org/downloads/pcg-c-basic-0.9.zip
void pcg32_srandom_r(pcg32_random_t *rng, uint64_t initstate, uint64_t initseq) {
	rng->state = 0U;
	rng->inc = (initseq << 1u) | 1u;
	pcg32_random_r(rng);
	rng->state += initstate;
	pcg32_random_r(rng);
}

// Source from https://github.com/imneme/pcg-c-basic/blob/master/pcg_basic.c
// pcg32_boundedrand_r(rng, bound):
//     Generate a uniformly distributed number, r, where 0 <= r < bound
uint32_t pcg32_boundedrand_r(pcg32_random_t *rng, uint32_t bound) {
	// To avoid bias, we need to make the range of the RNG a multiple of
	// bound, which we do by dropping output less than a threshold.
	// A naive scheme to calculate the threshold would be to do
	//
	//     uint32_t threshold = 0x100000000ull % bound;
	//
	// but 64-bit div/mod is slower than 32-bit div/mod (especially on
	// 32-bit platforms).  In essence, we do
	//
	//     uint32_t threshold = (0x100000000ull-bound) % bound;
	//
	// because this version will calculate the same modulus, but the LHS
	// value is less than 2^32.
	uint32_t threshold = -bound % bound;

	// Uniformity guarantees that this loop will terminate.  In practice, it
	// should usually terminate quickly; on average (assuming all bounds are
	// equally likely), 82.25% of the time, we can expect it to require just
	// one iteration.  In the worst case, someone passes a bound of 2^31 + 1
	// (i.e., 2147483649), which invalidates almost 50% of the range.  In
	// practice, bounds are typically small and only a tiny amount of the range
	// is eliminated.
	for (;;) {
		uint32_t r = pcg32_random_r(rng);
		if (r >= threshold)
			return r % bound;
	}
}
#line 0

#line 1 "sfw/core/vector2.cpp"
/*************************************************************************/
/*  vector2.cpp                                                          */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

real_t Vector2::angle() const {
	return Math::atan2(y, x);
}

real_t Vector2::length() const {
	return Math::sqrt(x * x + y * y);
}

real_t Vector2::length_squared() const {
	return x * x + y * y;
}

void Vector2::normalize() {
	real_t l = x * x + y * y;
	if (l != 0) {
		l = Math::sqrt(l);
		x /= l;
		y /= l;
	}
}

Vector2 Vector2::normalized() const {
	Vector2 v = *this;
	v.normalize();
	return v;
}

bool Vector2::is_normalized() const {
	// use length_squared() instead of length() to avoid sqrt(), makes it more stringent.
	return Math::is_equal_approx(length_squared(), 1, (real_t)UNIT_EPSILON);
}

real_t Vector2::distance_to(const Vector2 &p_vector2) const {
	return Math::sqrt((x - p_vector2.x) * (x - p_vector2.x) + (y - p_vector2.y) * (y - p_vector2.y));
}

real_t Vector2::distance_squared_to(const Vector2 &p_vector2) const {
	return (x - p_vector2.x) * (x - p_vector2.x) + (y - p_vector2.y) * (y - p_vector2.y);
}

real_t Vector2::angle_to(const Vector2 &p_vector2) const {
	return Math::atan2(cross(p_vector2), dot(p_vector2));
}

real_t Vector2::angle_to_point(const Vector2 &p_vector2) const {
	return Math::atan2(y - p_vector2.y, x - p_vector2.x);
}

real_t Vector2::dot(const Vector2 &p_other) const {
	return x * p_other.x + y * p_other.y;
}

real_t Vector2::cross(const Vector2 &p_other) const {
	return x * p_other.y - y * p_other.x;
}

Vector2 Vector2::sign() const {
	return Vector2(SGN(x), SGN(y));
}

Vector2 Vector2::floor() const {
	return Vector2(Math::floor(x), Math::floor(y));
}

Vector2 Vector2::ceil() const {
	return Vector2(Math::ceil(x), Math::ceil(y));
}

Vector2 Vector2::round() const {
	return Vector2(Math::round(x), Math::round(y));
}

Vector2 Vector2::rotated(real_t p_by) const {
	Vector2 v;
	v.set_rotation(angle() + p_by);
	v *= length();
	return v;
}

Vector2 Vector2::posmod(const real_t p_mod) const {
	return Vector2(Math::fposmod(x, p_mod), Math::fposmod(y, p_mod));
}

Vector2 Vector2::posmodv(const Vector2 &p_modv) const {
	return Vector2(Math::fposmod(x, p_modv.x), Math::fposmod(y, p_modv.y));
}

Vector2 Vector2::project(const Vector2 &p_to) const {
	return p_to * (dot(p_to) / p_to.length_squared());
}

Vector2 Vector2::snapped(const Vector2 &p_by) const {
	return Vector2(
			Math::stepify(x, p_by.x),
			Math::stepify(y, p_by.y));
}

Vector2 Vector2::limit_length(const real_t p_len) const {
	const real_t l = length();
	Vector2 v = *this;
	if (l > 0 && p_len < l) {
		v /= l;
		v *= p_len;
	}

	return v;
}

Vector2 Vector2::move_toward(const Vector2 &p_to, const real_t p_delta) const {
	Vector2 v = *this;
	Vector2 vd = p_to - v;
	real_t len = vd.length();
	return len <= p_delta || len < (real_t)CMP_EPSILON ? p_to : v + vd / len * p_delta;
}

// slide returns the component of the vector along the given plane, specified by its normal vector.
Vector2 Vector2::slide(const Vector2 &p_normal) const {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_V_MSG(!p_normal.is_normalized(), Vector2(), "The normal Vector2 must be normalized.");
#endif
	return *this - p_normal * this->dot(p_normal);
}

Vector2 Vector2::bounce(const Vector2 &p_normal) const {
	return -reflect(p_normal);
}

Vector2 Vector2::reflect(const Vector2 &p_normal) const {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_V_MSG(!p_normal.is_normalized(), Vector2(), "The normal Vector2 must be normalized.");
#endif
	return 2 * p_normal * this->dot(p_normal) - *this;
}

bool Vector2::is_equal_approx(const Vector2 &p_v) const {
	return Math::is_equal_approx(x, p_v.x) && Math::is_equal_approx(y, p_v.y);
}

Vector2::operator String() const {
	return "(" + String::num_real(x) + ", " + String::num_real(y) + ")";
}
#line 0

#line 1 "sfw/core/basis.cpp"
/*************************************************************************/
/*  basis.cpp                                                            */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

#define cofac(row1, col1, row2, col2) \
	(rows[row1][col1] * rows[row2][col2] - rows[row1][col2] * rows[row2][col1])

void Basis::from_z(const Vector3 &p_z) {
	if (Math::abs(p_z.z) > (real_t)Math_SQRT12) {
		// choose p in y-z plane
		real_t a = p_z[1] * p_z[1] + p_z[2] * p_z[2];
		real_t k = 1 / Math::sqrt(a);
		rows[0] = Vector3(0, -p_z[2] * k, p_z[1] * k);
		rows[1] = Vector3(a * k, -p_z[0] * rows[0][2], p_z[0] * rows[0][1]);
	} else {
		// choose p in x-y plane
		real_t a = p_z.x * p_z.x + p_z.y * p_z.y;
		real_t k = 1 / Math::sqrt(a);
		rows[0] = Vector3(-p_z.y * k, p_z.x * k, 0);
		rows[1] = Vector3(-p_z.z * rows[0].y, p_z.z * rows[0].x, a * k);
	}
	rows[2] = p_z;
}

void Basis::invert() {
	real_t co[3] = {
		cofac(1, 1, 2, 2), cofac(1, 2, 2, 0), cofac(1, 0, 2, 1)
	};
	real_t det = rows[0][0] * co[0] +
			rows[0][1] * co[1] +
			rows[0][2] * co[2];
#ifdef MATH_CHECKS
	ERR_FAIL_COND(det == 0);
#endif
	real_t s = 1 / det;

	set(co[0] * s, cofac(0, 2, 2, 1) * s, cofac(0, 1, 1, 2) * s,
			co[1] * s, cofac(0, 0, 2, 2) * s, cofac(0, 2, 1, 0) * s,
			co[2] * s, cofac(0, 1, 2, 0) * s, cofac(0, 0, 1, 1) * s);
}

void Basis::orthonormalize() {
	// Gram-Schmidt Process

	Vector3 x = get_column(0);
	Vector3 y = get_column(1);
	Vector3 z = get_column(2);

	x.normalize();
	y = (y - x * (x.dot(y)));
	y.normalize();
	z = (z - x * (x.dot(z)) - y * (y.dot(z)));
	z.normalize();

	set_column(0, x);
	set_column(1, y);
	set_column(2, z);
}

Basis Basis::orthonormalized() const {
	Basis c = *this;
	c.orthonormalize();
	return c;
}

bool Basis::is_orthogonal() const {
	Basis identity;
	Basis m = (*this) * transposed();

	return m.is_equal_approx(identity);
}

void Basis::orthogonalize() {
	Vector3 scl = get_scale();
	orthonormalize();
	scale_local(scl);
}

Basis Basis::orthogonalized() const {
	Basis c = *this;
	c.orthogonalize();
	return c;
}

bool Basis::is_diagonal() const {
	return (
			Math::is_zero_approx(rows[0][1]) && Math::is_zero_approx(rows[0][2]) &&
			Math::is_zero_approx(rows[1][0]) && Math::is_zero_approx(rows[1][2]) &&
			Math::is_zero_approx(rows[2][0]) && Math::is_zero_approx(rows[2][1]));
}

bool Basis::is_rotation() const {
	return Math::is_equal_approx(determinant(), 1, (real_t)UNIT_EPSILON) && is_orthogonal();
}

bool Basis::is_symmetric() const {
	if (!Math::is_equal_approx_ratio(rows[0][1], rows[1][0], (real_t)UNIT_EPSILON)) {
		return false;
	}
	if (!Math::is_equal_approx_ratio(rows[0][2], rows[2][0], (real_t)UNIT_EPSILON)) {
		return false;
	}
	if (!Math::is_equal_approx_ratio(rows[1][2], rows[2][1], (real_t)UNIT_EPSILON)) {
		return false;
	}

	return true;
}

Basis Basis::diagonalize() {
//NOTE: only implemented for symmetric matrices
//with the Jacobi iterative method method
#ifdef MATH_CHECKS
	ERR_FAIL_COND_V(!is_symmetric(), Basis());
#endif
	const int ite_max = 1024;

	real_t off_matrix_norm_2 = rows[0][1] * rows[0][1] + rows[0][2] * rows[0][2] + rows[1][2] * rows[1][2];

	int ite = 0;
	Basis acc_rot;
	while (off_matrix_norm_2 > (real_t)CMP_EPSILON2 && ite++ < ite_max) {
		real_t el01_2 = rows[0][1] * rows[0][1];
		real_t el02_2 = rows[0][2] * rows[0][2];
		real_t el12_2 = rows[1][2] * rows[1][2];
		// Find the pivot element
		int i, j;
		if (el01_2 > el02_2) {
			if (el12_2 > el01_2) {
				i = 1;
				j = 2;
			} else {
				i = 0;
				j = 1;
			}
		} else {
			if (el12_2 > el02_2) {
				i = 1;
				j = 2;
			} else {
				i = 0;
				j = 2;
			}
		}

		// Compute the rotation angle
		real_t angle;
		if (Math::is_equal_approx(rows[j][j], rows[i][i])) {
			angle = Math_PI / 4;
		} else {
			angle = 0.5f * Math::atan(2 * rows[i][j] / (rows[j][j] - rows[i][i]));
		}

		// Compute the rotation matrix
		Basis rot;
		rot.rows[i][i] = rot.rows[j][j] = Math::cos(angle);
		rot.rows[i][j] = -(rot.rows[j][i] = Math::sin(angle));

		// Update the off matrix norm
		off_matrix_norm_2 -= rows[i][j] * rows[i][j];

		// Apply the rotation
		*this = rot * *this * rot.transposed();
		acc_rot = rot * acc_rot;
	}

	return acc_rot;
}

Basis Basis::inverse() const {
	Basis inv = *this;
	inv.invert();
	return inv;
}

void Basis::transpose() {
	SWAP(rows[0][1], rows[1][0]);
	SWAP(rows[0][2], rows[2][0]);
	SWAP(rows[1][2], rows[2][1]);
}

Basis Basis::transposed() const {
	Basis tr = *this;
	tr.transpose();
	return tr;
}

Basis Basis::from_scale(const Vector3 &p_scale) {
	return Basis(p_scale.x, 0, 0, 0, p_scale.y, 0, 0, 0, p_scale.z);
}

// Multiplies the matrix from left by the scaling matrix: M -> S.M
// See the comment for Basis::rotated for further explanation.
void Basis::scale(const Vector3 &p_scale) {
	rows[0][0] *= p_scale.x;
	rows[0][1] *= p_scale.x;
	rows[0][2] *= p_scale.x;
	rows[1][0] *= p_scale.y;
	rows[1][1] *= p_scale.y;
	rows[1][2] *= p_scale.y;
	rows[2][0] *= p_scale.z;
	rows[2][1] *= p_scale.z;
	rows[2][2] *= p_scale.z;
}

Basis Basis::scaled(const Vector3 &p_scale) const {
	Basis m = *this;
	m.scale(p_scale);
	return m;
}

void Basis::scale_local(const Vector3 &p_scale) {
	// performs a scaling in object-local coordinate system:
	// M -> (M.S.Minv).M = M.S.
	*this = scaled_local(p_scale);
}

Basis Basis::scaled_local(const Vector3 &p_scale) const {
	Basis b;
	b.set_diagonal(p_scale);

	return (*this) * b;
}

void Basis::scale_orthogonal(const Vector3 &p_scale) {
	*this = scaled_orthogonal(p_scale);
}

Basis Basis::scaled_orthogonal(const Vector3 &p_scale) const {
	Basis m = *this;
	Vector3 s = Vector3(-1, -1, -1) + p_scale;
	Vector3 dots;
	Basis b;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			dots[j] += s[i] * Math::abs(m.get_column(i).normalized().dot(b.get_column(j)));
		}
	}
	m.scale_local(Vector3(1, 1, 1) + dots);
	return m;
}

real_t Basis::get_uniform_scale() const {
	return (rows[0].length() + rows[1].length() + rows[2].length()) / 3.0f;
}

void Basis::make_scale_uniform() {
	float l = (rows[0].length() + rows[1].length() + rows[2].length()) / 3.0f;
	for (int i = 0; i < 3; i++) {
		rows[i].normalize();
		rows[i] *= l;
	}
}

Vector3 Basis::get_scale_abs() const {
	return Vector3(
			Vector3(rows[0][0], rows[1][0], rows[2][0]).length(),
			Vector3(rows[0][1], rows[1][1], rows[2][1]).length(),
			Vector3(rows[0][2], rows[1][2], rows[2][2]).length());
}

Vector3 Basis::get_scale_local() const {
	real_t det_sign = SGN(determinant());
	return det_sign * Vector3(rows[0].length(), rows[1].length(), rows[2].length());
}

// get_scale works with get_rotation, use get_scale_abs if you need to enforce positive signature.
Vector3 Basis::get_scale() const {
	// FIXME: We are assuming M = R.S (R is rotation and S is scaling), and use polar decomposition to extract R and S.
	// A polar decomposition is M = O.P, where O is an orthogonal matrix (meaning rotation and reflection) and
	// P is a positive semi-definite matrix (meaning it contains absolute values of scaling along its diagonal).
	//
	// Despite being different from what we want to achieve, we can nevertheless make use of polar decomposition
	// here as follows. We can split O into a rotation and a reflection as O = R.Q, and obtain M = R.S where
	// we defined S = Q.P. Now, R is a proper rotation matrix and S is a (signed) scaling matrix,
	// which can involve negative scalings. However, there is a catch: unlike the polar decomposition of M = O.P,
	// the decomposition of O into a rotation and reflection matrix as O = R.Q is not unique.
	// Therefore, we are going to do this decomposition by sticking to a particular convention.
	// This may lead to confusion for some users though.
	//
	// The convention we use here is to absorb the sign flip into the scaling matrix.
	// The same convention is also used in other similar functions such as get_rotation_axis_angle, get_rotation, ...
	//
	// A proper way to get rid of this issue would be to store the scaling values (or at least their signs)
	// as a part of Basis. However, if we go that path, we need to disable direct (write) access to the
	// matrix rows.
	//
	// The rotation part of this decomposition is returned by get_rotation* functions.
	real_t det_sign = SGN(determinant());
	return det_sign * get_scale_abs();
}

// Decomposes a Basis into a rotation-reflection matrix (an element of the group O(3)) and a positive scaling matrix as B = O.S.
// Returns the rotation-reflection matrix via reference argument, and scaling information is returned as a Vector3.
// This (internal) function is too specific and named too ugly to expose to users, and probably there's no need to do so.
Vector3 Basis::rotref_posscale_decomposition(Basis &rotref) const {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_V(determinant() == 0, Vector3());

	Basis m = transposed() * (*this);
	ERR_FAIL_COND_V(!m.is_diagonal(), Vector3());
#endif
	Vector3 scale = get_scale();
	Basis inv_scale = Basis().scaled(scale.inverse()); // this will also absorb the sign of scale
	rotref = (*this) * inv_scale;

#ifdef MATH_CHECKS
	ERR_FAIL_COND_V(!rotref.is_orthogonal(), Vector3());
#endif
	return scale.abs();
}

// Multiplies the matrix from left by the rotation matrix: M -> R.M
// Note that this does *not* rotate the matrix itself.
//
// The main use of Basis is as Transform.basis, which is used a the transformation matrix
// of 3D object. Rotate here refers to rotation of the object (which is R * (*this)),
// not the matrix itself (which is R * (*this) * R.transposed()).
Basis Basis::rotated(const Vector3 &p_axis, real_t p_phi) const {
	return Basis(p_axis, p_phi) * (*this);
}

void Basis::rotate(const Vector3 &p_axis, real_t p_phi) {
	*this = rotated(p_axis, p_phi);
}

void Basis::rotate_local(const Vector3 &p_axis, real_t p_phi) {
	// performs a rotation in object-local coordinate system:
	// M -> (M.R.Minv).M = M.R.
	*this = rotated_local(p_axis, p_phi);
}
Basis Basis::rotated_local(const Vector3 &p_axis, real_t p_phi) const {
	return (*this) * Basis(p_axis, p_phi);
}

Basis Basis::rotated(const Vector3 &p_euler) const {
	return Basis(p_euler) * (*this);
}

void Basis::rotate(const Vector3 &p_euler) {
	*this = rotated(p_euler);
}

Basis Basis::rotated(const Quaternion &p_quat) const {
	return Basis(p_quat) * (*this);
}

void Basis::rotate(const Quaternion &p_quat) {
	*this = rotated(p_quat);
}

Vector3 Basis::get_rotation_euler() const {
	// Assumes that the matrix can be decomposed into a proper rotation and scaling matrix as M = R.S,
	// and returns the Euler angles corresponding to the rotation part, complementing get_scale().
	// See the comment in get_scale() for further information.
	Basis m = orthonormalized();
	real_t det = m.determinant();
	if (det < 0) {
		// Ensure that the determinant is 1, such that result is a proper rotation matrix which can be represented by Euler angles.
		m.scale(Vector3(-1, -1, -1));
	}

	return m.get_euler();
}

Quaternion Basis::get_rotation_quaternion() const {
	// Assumes that the matrix can be decomposed into a proper rotation and scaling matrix as M = R.S,
	// and returns the Euler angles corresponding to the rotation part, complementing get_scale().
	// See the comment in get_scale() for further information.
	Basis m = orthonormalized();
	real_t det = m.determinant();
	if (det < 0) {
		// Ensure that the determinant is 1, such that result is a proper rotation matrix which can be represented by Euler angles.
		m.scale(Vector3(-1, -1, -1));
	}

	return m.get_quaternion();
}

void Basis::rotate_to_align(const Vector3 &p_start_direction, const Vector3 &p_end_direction) {
	// Takes two vectors and rotates the basis from the first vector to the second vector.
	// Adopted from: https://gist.github.com/kevinmoran/b45980723e53edeb8a5a43c49f134724
	const Vector3 axis = p_start_direction.cross(p_end_direction).normalized();
	if (axis.length_squared() != 0) {
		real_t dot = p_start_direction.dot(p_end_direction);
		dot = CLAMP(dot, -1.0, 1.0);
		const real_t angle_rads = Math::acos(dot);
		*this = Basis(axis, angle_rads) * (*this);
	}
}

void Basis::get_rotation_axis_angle(Vector3 &p_axis, real_t &p_angle) const {
	// Assumes that the matrix can be decomposed into a proper rotation and scaling matrix as M = R.S,
	// and returns the Euler angles corresponding to the rotation part, complementing get_scale().
	// See the comment in get_scale() for further information.
	Basis m = orthonormalized();
	real_t det = m.determinant();
	if (det < 0) {
		// Ensure that the determinant is 1, such that result is a proper rotation matrix which can be represented by Euler angles.
		m.scale(Vector3(-1, -1, -1));
	}

	m.get_axis_angle(p_axis, p_angle);
}

void Basis::get_rotation_axis_angle_local(Vector3 &p_axis, real_t &p_angle) const {
	// Assumes that the matrix can be decomposed into a proper rotation and scaling matrix as M = R.S,
	// and returns the Euler angles corresponding to the rotation part, complementing get_scale().
	// See the comment in get_scale() for further information.
	Basis m = transposed();
	m.orthonormalize();
	real_t det = m.determinant();
	if (det < 0) {
		// Ensure that the determinant is 1, such that result is a proper rotation matrix which can be represented by Euler angles.
		m.scale(Vector3(-1, -1, -1));
	}

	m.get_axis_angle(p_axis, p_angle);
	p_angle = -p_angle;
}

// get_euler_xyz returns a vector containing the Euler angles in the format
// (a1,a2,a3), where a3 is the angle of the first rotation, and a1 is the last
// (following the convention they are commonly defined in the literature).
//
// The current implementation uses XYZ convention (Z is the first rotation),
// so euler.z is the angle of the (first) rotation around Z axis and so on,
//
// And thus, assuming the matrix is a rotation matrix, this function returns
// the angles in the decomposition R = X(a1).Y(a2).Z(a3) where Z(a) rotates
// around the z-axis by a and so on.
Vector3 Basis::get_euler_xyz() const {
	// Euler angles in XYZ convention.
	// See https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
	//
	// rot =  cy*cz          -cy*sz           sy
	//        cz*sx*sy+cx*sz  cx*cz-sx*sy*sz -cy*sx
	//       -cx*cz*sy+sx*sz  cz*sx+cx*sy*sz  cx*cy

	Vector3 euler;
	real_t sy = rows[0][2];
	if (sy < (1 - (real_t)CMP_EPSILON)) {
		if (sy > -(1 - (real_t)CMP_EPSILON)) {
			// is this a pure Y rotation?
			if (rows[1][0] == 0 && rows[0][1] == 0 && rows[1][2] == 0 && rows[2][1] == 0 && rows[1][1] == 1) {
				// return the simplest form (human friendlier in editor and scripts)
				euler.x = 0;
				euler.y = atan2(rows[0][2], rows[0][0]);
				euler.z = 0;
			} else {
				euler.x = Math::atan2(-rows[1][2], rows[2][2]);
				euler.y = Math::asin(sy);
				euler.z = Math::atan2(-rows[0][1], rows[0][0]);
			}
		} else {
			euler.x = Math::atan2(rows[2][1], rows[1][1]);
			euler.y = -Math_PI / 2.0;
			euler.z = 0.0;
		}
	} else {
		euler.x = Math::atan2(rows[2][1], rows[1][1]);
		euler.y = Math_PI / 2.0;
		euler.z = 0.0;
	}
	return euler;
}

// set_euler_xyz expects a vector containing the Euler angles in the format
// (ax,ay,az), where ax is the angle of rotation around x axis,
// and similar for other axes.
// The current implementation uses XYZ convention (Z is the first rotation).
void Basis::set_euler_xyz(const Vector3 &p_euler) {
	real_t c, s;

	c = Math::cos(p_euler.x);
	s = Math::sin(p_euler.x);
	Basis xmat(1, 0, 0, 0, c, -s, 0, s, c);

	c = Math::cos(p_euler.y);
	s = Math::sin(p_euler.y);
	Basis ymat(c, 0, s, 0, 1, 0, -s, 0, c);

	c = Math::cos(p_euler.z);
	s = Math::sin(p_euler.z);
	Basis zmat(c, -s, 0, s, c, 0, 0, 0, 1);

	//optimizer will optimize away all this anyway
	*this = xmat * (ymat * zmat);
}

Vector3 Basis::get_euler_xzy() const {
	// Euler angles in XZY convention.
	// See https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
	//
	// rot =  cz*cy             -sz             cz*sy
	//        sx*sy+cx*cy*sz    cx*cz           cx*sz*sy-cy*sx
	//        cy*sx*sz          cz*sx           cx*cy+sx*sz*sy

	Vector3 euler;
	real_t sz = rows[0][1];
	if (sz < (1 - (real_t)CMP_EPSILON)) {
		if (sz > -(1 - (real_t)CMP_EPSILON)) {
			euler.x = Math::atan2(rows[2][1], rows[1][1]);
			euler.y = Math::atan2(rows[0][2], rows[0][0]);
			euler.z = Math::asin(-sz);
		} else {
			// It's -1
			euler.x = -Math::atan2(rows[1][2], rows[2][2]);
			euler.y = 0.0;
			euler.z = Math_PI / 2.0;
		}
	} else {
		// It's 1
		euler.x = -Math::atan2(rows[1][2], rows[2][2]);
		euler.y = 0.0;
		euler.z = -Math_PI / 2.0;
	}
	return euler;
}

void Basis::set_euler_xzy(const Vector3 &p_euler) {
	real_t c, s;

	c = Math::cos(p_euler.x);
	s = Math::sin(p_euler.x);
	Basis xmat(1, 0, 0, 0, c, -s, 0, s, c);

	c = Math::cos(p_euler.y);
	s = Math::sin(p_euler.y);
	Basis ymat(c, 0, s, 0, 1, 0, -s, 0, c);

	c = Math::cos(p_euler.z);
	s = Math::sin(p_euler.z);
	Basis zmat(c, -s, 0, s, c, 0, 0, 0, 1);

	*this = xmat * zmat * ymat;
}

Vector3 Basis::get_euler_yzx() const {
	// Euler angles in YZX convention.
	// See https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
	//
	// rot =  cy*cz             sy*sx-cy*cx*sz     cx*sy+cy*sz*sx
	//        sz                cz*cx              -cz*sx
	//        -cz*sy            cy*sx+cx*sy*sz     cy*cx-sy*sz*sx

	Vector3 euler;
	real_t sz = rows[1][0];
	if (sz < (1 - (real_t)CMP_EPSILON)) {
		if (sz > -(1 - (real_t)CMP_EPSILON)) {
			euler.x = Math::atan2(-rows[1][2], rows[1][1]);
			euler.y = Math::atan2(-rows[2][0], rows[0][0]);
			euler.z = Math::asin(sz);
		} else {
			// It's -1
			euler.x = Math::atan2(rows[2][1], rows[2][2]);
			euler.y = 0.0;
			euler.z = -Math_PI / 2.0;
		}
	} else {
		// It's 1
		euler.x = Math::atan2(rows[2][1], rows[2][2]);
		euler.y = 0.0;
		euler.z = Math_PI / 2.0;
	}
	return euler;
}

void Basis::set_euler_yzx(const Vector3 &p_euler) {
	real_t c, s;

	c = Math::cos(p_euler.x);
	s = Math::sin(p_euler.x);
	Basis xmat(1, 0, 0, 0, c, -s, 0, s, c);

	c = Math::cos(p_euler.y);
	s = Math::sin(p_euler.y);
	Basis ymat(c, 0, s, 0, 1, 0, -s, 0, c);

	c = Math::cos(p_euler.z);
	s = Math::sin(p_euler.z);
	Basis zmat(c, -s, 0, s, c, 0, 0, 0, 1);

	*this = ymat * zmat * xmat;
}

// get_euler_yxz returns a vector containing the Euler angles in the YXZ convention,
// as in first-Z, then-X, last-Y. The angles for X, Y, and Z rotations are returned
// as the x, y, and z components of a Vector3 respectively.
Vector3 Basis::get_euler_yxz() const {
	// Euler angles in YXZ convention.
	// See https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
	//
	// rot =  cy*cz+sy*sx*sz    cz*sy*sx-cy*sz        cx*sy
	//        cx*sz             cx*cz                 -sx
	//        cy*sx*sz-cz*sy    cy*cz*sx+sy*sz        cy*cx

	Vector3 euler;

	real_t m12 = rows[1][2];

	if (m12 < (1 - (real_t)CMP_EPSILON)) {
		if (m12 > -(1 - (real_t)CMP_EPSILON)) {
			// is this a pure X rotation?
			if (rows[1][0] == 0 && rows[0][1] == 0 && rows[0][2] == 0 && rows[2][0] == 0 && rows[0][0] == 1) {
				// return the simplest form (human friendlier in editor and scripts)
				euler.x = atan2(-m12, rows[1][1]);
				euler.y = 0;
				euler.z = 0;
			} else {
				euler.x = asin(-m12);
				euler.y = atan2(rows[0][2], rows[2][2]);
				euler.z = atan2(rows[1][0], rows[1][1]);
			}
		} else { // m12 == -1
			euler.x = Math_PI * 0.5;
			euler.y = atan2(rows[0][1], rows[0][0]);
			euler.z = 0;
		}
	} else { // m12 == 1
		euler.x = -Math_PI * 0.5;
		euler.y = -atan2(rows[0][1], rows[0][0]);
		euler.z = 0;
	}

	return euler;
}

// set_euler_yxz expects a vector containing the Euler angles in the format
// (ax,ay,az), where ax is the angle of rotation around x axis,
// and similar for other axes.
// The current implementation uses YXZ convention (Z is the first rotation).
void Basis::set_euler_yxz(const Vector3 &p_euler) {
	real_t c, s;

	c = Math::cos(p_euler.x);
	s = Math::sin(p_euler.x);
	Basis xmat(1, 0, 0, 0, c, -s, 0, s, c);

	c = Math::cos(p_euler.y);
	s = Math::sin(p_euler.y);
	Basis ymat(c, 0, s, 0, 1, 0, -s, 0, c);

	c = Math::cos(p_euler.z);
	s = Math::sin(p_euler.z);
	Basis zmat(c, -s, 0, s, c, 0, 0, 0, 1);

	//optimizer will optimize away all this anyway
	*this = ymat * xmat * zmat;
}

Vector3 Basis::get_euler_zxy() const {
	// Euler angles in ZXY convention.
	// See https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
	//
	// rot =  cz*cy-sz*sx*sy    -cx*sz                cz*sy+cy*sz*sx
	//        cy*sz+cz*sx*sy    cz*cx                 sz*sy-cz*cy*sx
	//        -cx*sy            sx                    cx*cy
	Vector3 euler;
	real_t sx = rows[2][1];
	if (sx < (1 - (real_t)CMP_EPSILON)) {
		if (sx > -(1 - (real_t)CMP_EPSILON)) {
			euler.x = Math::asin(sx);
			euler.y = Math::atan2(-rows[2][0], rows[2][2]);
			euler.z = Math::atan2(-rows[0][1], rows[1][1]);
		} else {
			// It's -1
			euler.x = -Math_PI / 2.0;
			euler.y = Math::atan2(rows[0][2], rows[0][0]);
			euler.z = 0;
		}
	} else {
		// It's 1
		euler.x = Math_PI / 2.0;
		euler.y = Math::atan2(rows[0][2], rows[0][0]);
		euler.z = 0;
	}
	return euler;
}

void Basis::set_euler_zxy(const Vector3 &p_euler) {
	real_t c, s;

	c = Math::cos(p_euler.x);
	s = Math::sin(p_euler.x);
	Basis xmat(1, 0, 0, 0, c, -s, 0, s, c);

	c = Math::cos(p_euler.y);
	s = Math::sin(p_euler.y);
	Basis ymat(c, 0, s, 0, 1, 0, -s, 0, c);

	c = Math::cos(p_euler.z);
	s = Math::sin(p_euler.z);
	Basis zmat(c, -s, 0, s, c, 0, 0, 0, 1);

	*this = zmat * xmat * ymat;
}

Vector3 Basis::get_euler_zyx() const {
	// Euler angles in ZYX convention.
	// See https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
	//
	// rot =  cz*cy             cz*sy*sx-cx*sz        sz*sx+cz*cx*cy
	//        cy*sz             cz*cx+sz*sy*sx        cx*sz*sy-cz*sx
	//        -sy               cy*sx                 cy*cx
	Vector3 euler;
	real_t sy = rows[2][0];
	if (sy < (1 - (real_t)CMP_EPSILON)) {
		if (sy > -(1 - (real_t)CMP_EPSILON)) {
			euler.x = Math::atan2(rows[2][1], rows[2][2]);
			euler.y = Math::asin(-sy);
			euler.z = Math::atan2(rows[1][0], rows[0][0]);
		} else {
			// It's -1
			euler.x = 0;
			euler.y = Math_PI / 2.0;
			euler.z = -Math::atan2(rows[0][1], rows[1][1]);
		}
	} else {
		// It's 1
		euler.x = 0;
		euler.y = -Math_PI / 2.0;
		euler.z = -Math::atan2(rows[0][1], rows[1][1]);
	}
	return euler;
}

void Basis::set_euler_zyx(const Vector3 &p_euler) {
	real_t c, s;

	c = Math::cos(p_euler.x);
	s = Math::sin(p_euler.x);
	Basis xmat(1, 0, 0, 0, c, -s, 0, s, c);

	c = Math::cos(p_euler.y);
	s = Math::sin(p_euler.y);
	Basis ymat(c, 0, s, 0, 1, 0, -s, 0, c);

	c = Math::cos(p_euler.z);
	s = Math::sin(p_euler.z);
	Basis zmat(c, -s, 0, s, c, 0, 0, 0, 1);

	*this = zmat * ymat * xmat;
}

bool Basis::is_equal_approx(const Basis &p_basis) const {
	return rows[0].is_equal_approx(p_basis.rows[0]) && rows[1].is_equal_approx(p_basis.rows[1]) && rows[2].is_equal_approx(p_basis.rows[2]);
}

bool Basis::is_equal_approx_ratio(const Basis &a, const Basis &b, real_t p_epsilon) const {
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (!Math::is_equal_approx_ratio(a.rows[i][j], b.rows[i][j], p_epsilon)) {
				return false;
			}
		}
	}

	return true;
}

bool Basis::operator==(const Basis &p_matrix) const {
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (rows[i][j] != p_matrix.rows[i][j]) {
				return false;
			}
		}
	}

	return true;
}

bool Basis::operator!=(const Basis &p_matrix) const {
	return (!(*this == p_matrix));
}

Basis::operator String() const {
	return "[X: " + get_axis(0).operator String() +
			", Y: " + get_axis(1).operator String() +
			", Z: " + get_axis(2).operator String() + "]";
}

Quaternion Basis::get_quaternion() const {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_V_MSG(!is_rotation(), Quaternion(), "Basis must be normalized in order to be casted to a Quaternion. Use get_rotation_quaternion() or call orthonormalized() if the Basis contains linearly independent vectors.");
#endif
	/* Allow getting a quaternion from an unnormalized transform */
	Basis m = *this;
	real_t trace = m.rows[0][0] + m.rows[1][1] + m.rows[2][2];
	real_t temp[4];

	if (trace > 0) {
		real_t s = Math::sqrt(trace + 1);
		temp[3] = (s * 0.5f);
		s = 0.5f / s;

		temp[0] = ((m.rows[2][1] - m.rows[1][2]) * s);
		temp[1] = ((m.rows[0][2] - m.rows[2][0]) * s);
		temp[2] = ((m.rows[1][0] - m.rows[0][1]) * s);
	} else {
		int i = m.rows[0][0] < m.rows[1][1]
				? (m.rows[1][1] < m.rows[2][2] ? 2 : 1)
				: (m.rows[0][0] < m.rows[2][2] ? 2 : 0);
		int j = (i + 1) % 3;
		int k = (i + 2) % 3;

		real_t s = Math::sqrt(m.rows[i][i] - m.rows[j][j] - m.rows[k][k] + 1);
		temp[i] = s * 0.5f;
		s = 0.5f / s;

		temp[3] = (m.rows[k][j] - m.rows[j][k]) * s;
		temp[j] = (m.rows[j][i] + m.rows[i][j]) * s;
		temp[k] = (m.rows[k][i] + m.rows[i][k]) * s;
	}

	return Quaternion(temp[0], temp[1], temp[2], temp[3]);
}

static const Basis _ortho_bases[24] = {
	Basis(1, 0, 0, 0, 1, 0, 0, 0, 1),
	Basis(0, -1, 0, 1, 0, 0, 0, 0, 1),
	Basis(-1, 0, 0, 0, -1, 0, 0, 0, 1),
	Basis(0, 1, 0, -1, 0, 0, 0, 0, 1),
	Basis(1, 0, 0, 0, 0, -1, 0, 1, 0),
	Basis(0, 0, 1, 1, 0, 0, 0, 1, 0),
	Basis(-1, 0, 0, 0, 0, 1, 0, 1, 0),
	Basis(0, 0, -1, -1, 0, 0, 0, 1, 0),
	Basis(1, 0, 0, 0, -1, 0, 0, 0, -1),
	Basis(0, 1, 0, 1, 0, 0, 0, 0, -1),
	Basis(-1, 0, 0, 0, 1, 0, 0, 0, -1),
	Basis(0, -1, 0, -1, 0, 0, 0, 0, -1),
	Basis(1, 0, 0, 0, 0, 1, 0, -1, 0),
	Basis(0, 0, -1, 1, 0, 0, 0, -1, 0),
	Basis(-1, 0, 0, 0, 0, -1, 0, -1, 0),
	Basis(0, 0, 1, -1, 0, 0, 0, -1, 0),
	Basis(0, 0, 1, 0, 1, 0, -1, 0, 0),
	Basis(0, -1, 0, 0, 0, 1, -1, 0, 0),
	Basis(0, 0, -1, 0, -1, 0, -1, 0, 0),
	Basis(0, 1, 0, 0, 0, -1, -1, 0, 0),
	Basis(0, 0, 1, 0, -1, 0, 1, 0, 0),
	Basis(0, 1, 0, 0, 0, 1, 1, 0, 0),
	Basis(0, 0, -1, 0, 1, 0, 1, 0, 0),
	Basis(0, -1, 0, 0, 0, -1, 1, 0, 0)
};

int Basis::get_orthogonal_index() const {
	//could be sped up if i come up with a way
	Basis orth = *this;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			real_t v = orth[i][j];
			if (v > 0.5f) {
				v = 1;
			} else if (v < -0.5f) {
				v = -1;
			} else {
				v = 0;
			}

			orth[i][j] = v;
		}
	}

	for (int i = 0; i < 24; i++) {
		if (_ortho_bases[i] == orth) {
			return i;
		}
	}

	return 0;
}

void Basis::set_orthogonal_index(int p_index) {
	//there only exist 24 orthogonal bases in r3
	ERR_FAIL_INDEX(p_index, 24);

	*this = _ortho_bases[p_index];
}

void Basis::get_axis_angle(Vector3 &r_axis, real_t &r_angle) const {
	/* checking this is a bad idea, because obtaining from scaled transform is a valid use case
#ifdef MATH_CHECKS
	ERR_FAIL_COND(!is_rotation());
#endif
	*/

	// https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/index.htm
	real_t x, y, z; // Variables for result.
	if (Math::is_zero_approx(rows[0][1] - rows[1][0]) && Math::is_zero_approx(rows[0][2] - rows[2][0]) && Math::is_zero_approx(rows[1][2] - rows[2][1])) {
		// Singularity found.
		// First check for identity matrix which must have +1 for all terms in leading diagonal and zero in other terms.
		if (is_diagonal() && (Math::abs(rows[0][0] + rows[1][1] + rows[2][2] - 3) < 3 * CMP_EPSILON)) {
			// This singularity is identity matrix so angle = 0.

			r_axis = Vector3(0, 1, 0);
			r_angle = 0;
			return;
		}

		// Otherwise this singularity is angle = 180
		real_t xx = (rows[0][0] + 1) / 2;
		real_t yy = (rows[1][1] + 1) / 2;
		real_t zz = (rows[2][2] + 1) / 2;
		real_t xy = (rows[0][1] + rows[1][0]) / 4;
		real_t xz = (rows[0][2] + rows[2][0]) / 4;
		real_t yz = (rows[1][2] + rows[2][1]) / 4;

		if ((xx > yy) && (xx > zz)) { // rows[0][0] is the largest diagonal term.
			if (xx < CMP_EPSILON) {
				x = 0;
				y = Math_SQRT12;
				z = Math_SQRT12;
			} else {
				x = Math::sqrt(xx);
				y = xy / x;
				z = xz / x;
			}
		} else if (yy > zz) { // rows[1][1] is the largest diagonal term.
			if (yy < CMP_EPSILON) {
				x = Math_SQRT12;
				y = 0;
				z = Math_SQRT12;
			} else {
				y = Math::sqrt(yy);
				x = xy / y;
				z = yz / y;
			}
		} else { // rows[2][2] is the largest diagonal term so base result on this.
			if (zz < CMP_EPSILON) {
				x = Math_SQRT12;
				y = Math_SQRT12;
				z = 0;
			} else {
				z = Math::sqrt(zz);
				x = xz / z;
				y = yz / z;
			}
		}
		r_axis = Vector3(x, y, z);
		r_angle = Math_PI;
		return;
	}

	// As we have reached here there are no singularities so we can handle normally
	real_t s = Math::sqrt((rows[2][1] - rows[1][2]) * (rows[2][1] - rows[1][2]) + (rows[0][2] - rows[2][0]) * (rows[0][2] - rows[2][0]) + (rows[1][0] - rows[0][1]) * (rows[1][0] - rows[0][1])); // Used to normalise.

	if (Math::abs(s) < CMP_EPSILON) {
		// Prevent divide by zero, should not happen if matrix is orthogonal and should be caught by singularity test above.
		s = 1;
	}

	x = (rows[2][1] - rows[1][2]) / s;
	y = (rows[0][2] - rows[2][0]) / s;
	z = (rows[1][0] - rows[0][1]) / s;

	r_axis = Vector3(x, y, z);
	// acos does clamping.
	r_angle = Math::acos((rows[0][0] + rows[1][1] + rows[2][2] - 1) / 2);
}

void Basis::set_quaternion(const Quaternion &p_quat) {
	real_t d = p_quat.length_squared();
	real_t s = 2 / d;
	real_t xs = p_quat.x * s, ys = p_quat.y * s, zs = p_quat.z * s;
	real_t wx = p_quat.w * xs, wy = p_quat.w * ys, wz = p_quat.w * zs;
	real_t xx = p_quat.x * xs, xy = p_quat.x * ys, xz = p_quat.x * zs;
	real_t yy = p_quat.y * ys, yz = p_quat.y * zs, zz = p_quat.z * zs;
	set(1 - (yy + zz), xy - wz, xz + wy,
			xy + wz, 1 - (xx + zz), yz - wx,
			xz - wy, yz + wx, 1 - (xx + yy));
}

void Basis::set_axis_angle(const Vector3 &p_axis, real_t p_phi) {
// Rotation matrix from axis and angle, see https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_angle
#ifdef MATH_CHECKS
	ERR_FAIL_COND_MSG(!p_axis.is_normalized(), "The axis Vector3 must be normalized.");
#endif
	Vector3 axis_sq(p_axis.x * p_axis.x, p_axis.y * p_axis.y, p_axis.z * p_axis.z);
	real_t cosine = Math::cos(p_phi);
	rows[0][0] = axis_sq.x + cosine * (1 - axis_sq.x);
	rows[1][1] = axis_sq.y + cosine * (1 - axis_sq.y);
	rows[2][2] = axis_sq.z + cosine * (1 - axis_sq.z);

	real_t sine = Math::sin(p_phi);
	real_t t = 1 - cosine;

	real_t xyzt = p_axis.x * p_axis.y * t;
	real_t zyxs = p_axis.z * sine;
	rows[0][1] = xyzt - zyxs;
	rows[1][0] = xyzt + zyxs;

	xyzt = p_axis.x * p_axis.z * t;
	zyxs = p_axis.y * sine;
	rows[0][2] = xyzt + zyxs;
	rows[2][0] = xyzt - zyxs;

	xyzt = p_axis.y * p_axis.z * t;
	zyxs = p_axis.x * sine;
	rows[1][2] = xyzt - zyxs;
	rows[2][1] = xyzt + zyxs;
}

void Basis::set_axis_angle_scale(const Vector3 &p_axis, real_t p_phi, const Vector3 &p_scale) {
	set_diagonal(p_scale);
	rotate(p_axis, p_phi);
}

void Basis::set_euler_scale(const Vector3 &p_euler, const Vector3 &p_scale) {
	set_diagonal(p_scale);
	rotate(p_euler);
}

void Basis::set_quaternion_scale(const Quaternion &p_quat, const Vector3 &p_scale) {
	set_diagonal(p_scale);
	rotate(p_quat);
}

void Basis::set_diagonal(const Vector3 &p_diag) {
	rows[0][0] = p_diag.x;
	rows[0][1] = 0;
	rows[0][2] = 0;

	rows[1][0] = 0;
	rows[1][1] = p_diag.y;
	rows[1][2] = 0;

	rows[2][0] = 0;
	rows[2][1] = 0;
	rows[2][2] = p_diag.z;
}

Basis Basis::slerp(const Basis &p_to, const real_t &p_weight) const {
	//consider scale
	Quaternion from(*this);
	Quaternion to(p_to);

	Basis b(from.slerp(to, p_weight));
	b.rows[0] *= Math::lerp(rows[0].length(), p_to.rows[0].length(), p_weight);
	b.rows[1] *= Math::lerp(rows[1].length(), p_to.rows[1].length(), p_weight);
	b.rows[2] *= Math::lerp(rows[2].length(), p_to.rows[2].length(), p_weight);

	return b;
}

void Basis::rotate_sh(real_t *p_values) {
	// code by John Hable
	// http://filmicworlds.com/blog/simple-and-fast-spherical-harmonic-rotation/
	// this code is Public Domain

	const static real_t s_c3 = 0.94617469575; // (3*sqrt(5))/(4*sqrt(pi))
	const static real_t s_c4 = -0.31539156525; // (-sqrt(5))/(4*sqrt(pi))
	const static real_t s_c5 = 0.54627421529; // (sqrt(15))/(4*sqrt(pi))

	const static real_t s_c_scale = 1.0 / 0.91529123286551084;
	const static real_t s_c_scale_inv = 0.91529123286551084;

	const static real_t s_rc2 = 1.5853309190550713 * s_c_scale;
	const static real_t s_c4_div_c3 = s_c4 / s_c3;
	const static real_t s_c4_div_c3_x2 = (s_c4 / s_c3) * 2.0;

	const static real_t s_scale_dst2 = s_c3 * s_c_scale_inv;
	const static real_t s_scale_dst4 = s_c5 * s_c_scale_inv;

	const real_t src[9] = { p_values[0], p_values[1], p_values[2], p_values[3], p_values[4], p_values[5], p_values[6], p_values[7], p_values[8] };

	real_t m00 = rows[0][0];
	real_t m01 = rows[0][1];
	real_t m02 = rows[0][2];
	real_t m10 = rows[1][0];
	real_t m11 = rows[1][1];
	real_t m12 = rows[1][2];
	real_t m20 = rows[2][0];
	real_t m21 = rows[2][1];
	real_t m22 = rows[2][2];

	p_values[0] = src[0];
	p_values[1] = m11 * src[1] - m12 * src[2] + m10 * src[3];
	p_values[2] = -m21 * src[1] + m22 * src[2] - m20 * src[3];
	p_values[3] = m01 * src[1] - m02 * src[2] + m00 * src[3];

	real_t sh0 = src[7] + src[8] + src[8] - src[5];
	real_t sh1 = src[4] + s_rc2 * src[6] + src[7] + src[8];
	real_t sh2 = src[4];
	real_t sh3 = -src[7];
	real_t sh4 = -src[5];

	// Rotations.  R0 and R1 just use the raw matrix columns
	real_t r2x = m00 + m01;
	real_t r2y = m10 + m11;
	real_t r2z = m20 + m21;

	real_t r3x = m00 + m02;
	real_t r3y = m10 + m12;
	real_t r3z = m20 + m22;

	real_t r4x = m01 + m02;
	real_t r4y = m11 + m12;
	real_t r4z = m21 + m22;

	// dense matrix multiplication one column at a time

	// column 0
	real_t sh0_x = sh0 * m00;
	real_t sh0_y = sh0 * m10;
	real_t d0 = sh0_x * m10;
	real_t d1 = sh0_y * m20;
	real_t d2 = sh0 * (m20 * m20 + s_c4_div_c3);
	real_t d3 = sh0_x * m20;
	real_t d4 = sh0_x * m00 - sh0_y * m10;

	// column 1
	real_t sh1_x = sh1 * m02;
	real_t sh1_y = sh1 * m12;
	d0 += sh1_x * m12;
	d1 += sh1_y * m22;
	d2 += sh1 * (m22 * m22 + s_c4_div_c3);
	d3 += sh1_x * m22;
	d4 += sh1_x * m02 - sh1_y * m12;

	// column 2
	real_t sh2_x = sh2 * r2x;
	real_t sh2_y = sh2 * r2y;
	d0 += sh2_x * r2y;
	d1 += sh2_y * r2z;
	d2 += sh2 * (r2z * r2z + s_c4_div_c3_x2);
	d3 += sh2_x * r2z;
	d4 += sh2_x * r2x - sh2_y * r2y;

	// column 3
	real_t sh3_x = sh3 * r3x;
	real_t sh3_y = sh3 * r3y;
	d0 += sh3_x * r3y;
	d1 += sh3_y * r3z;
	d2 += sh3 * (r3z * r3z + s_c4_div_c3_x2);
	d3 += sh3_x * r3z;
	d4 += sh3_x * r3x - sh3_y * r3y;

	// column 4
	real_t sh4_x = sh4 * r4x;
	real_t sh4_y = sh4 * r4y;
	d0 += sh4_x * r4y;
	d1 += sh4_y * r4z;
	d2 += sh4 * (r4z * r4z + s_c4_div_c3_x2);
	d3 += sh4_x * r4z;
	d4 += sh4_x * r4x - sh4_y * r4y;

	// extra multipliers
	p_values[4] = d0;
	p_values[5] = -d1;
	p_values[6] = d2 * s_scale_dst2;
	p_values[7] = -d3;
	p_values[8] = d4 * s_scale_dst4;
}

Basis Basis::looking_at(const Vector3 &p_target, const Vector3 &p_up) {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_V_MSG(p_target.is_equal_approx(Vector3()), Basis(), "The target vector can't be zero.");
	ERR_FAIL_COND_V_MSG(p_up.is_equal_approx(Vector3()), Basis(), "The up vector can't be zero.");
#endif
	Vector3 v_z = -p_target.normalized();
	Vector3 v_x = p_up.cross(v_z);
#ifdef MATH_CHECKS
	ERR_FAIL_COND_V_MSG(v_x.is_equal_approx(Vector3()), Basis(), "The target vector and up vector can't be parallel to each other.");
#endif
	v_x.normalize();
	Vector3 v_y = v_z.cross(v_x);

	Basis basis;
	basis.set_columns(v_x, v_y, v_z);
	return basis;
}

#undef cofac
#line 0

#line 1 "sfw/core/face3.cpp"
/*************************************************************************/
/*  face3.cpp                                                            */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

int Face3::split_by_plane(const Plane &p_plane, Face3 p_res[3], bool p_is_point_over[3]) const {
	ERR_FAIL_COND_V(is_degenerate(), 0);

	Vector3 above[4];
	int above_count = 0;

	Vector3 below[4];
	int below_count = 0;

	for (int i = 0; i < 3; i++) {
		if (p_plane.has_point(vertex[i], (real_t)CMP_EPSILON)) { // point is in plane

			ERR_FAIL_COND_V(above_count >= 4, 0);
			above[above_count++] = vertex[i];
			ERR_FAIL_COND_V(below_count >= 4, 0);
			below[below_count++] = vertex[i];

		} else {
			if (p_plane.is_point_over(vertex[i])) {
				//Point is over
				ERR_FAIL_COND_V(above_count >= 4, 0);
				above[above_count++] = vertex[i];

			} else {
				//Point is under
				ERR_FAIL_COND_V(below_count >= 4, 0);
				below[below_count++] = vertex[i];
			}

			/* Check for Intersection between this and the next vertex*/

			Vector3 inters;
			if (!p_plane.intersects_segment(vertex[i], vertex[(i + 1) % 3], &inters)) {
				continue;
			}

			/* Intersection goes to both */
			ERR_FAIL_COND_V(above_count >= 4, 0);
			above[above_count++] = inters;
			ERR_FAIL_COND_V(below_count >= 4, 0);
			below[below_count++] = inters;
		}
	}

	int polygons_created = 0;

	ERR_FAIL_COND_V(above_count >= 4 && below_count >= 4, 0); //bug in the algo

	if (above_count >= 3) {
		p_res[polygons_created] = Face3(above[0], above[1], above[2]);
		p_is_point_over[polygons_created] = true;
		polygons_created++;

		if (above_count == 4) {
			p_res[polygons_created] = Face3(above[2], above[3], above[0]);
			p_is_point_over[polygons_created] = true;
			polygons_created++;
		}
	}

	if (below_count >= 3) {
		p_res[polygons_created] = Face3(below[0], below[1], below[2]);
		p_is_point_over[polygons_created] = false;
		polygons_created++;

		if (below_count == 4) {
			p_res[polygons_created] = Face3(below[2], below[3], below[0]);
			p_is_point_over[polygons_created] = false;
			polygons_created++;
		}
	}

	return polygons_created;
}

bool Face3::intersects_ray(const Vector3 &p_from, const Vector3 &p_dir, Vector3 *p_intersection) const {
	//return Geometry::ray_intersects_triangle(p_from, p_dir, vertex[0], vertex[1], vertex[2], p_intersection);
	return false;
}

bool Face3::intersects_segment(const Vector3 &p_from, const Vector3 &p_dir, Vector3 *p_intersection) const {
	//return Geometry::segment_intersects_triangle(p_from, p_dir, vertex[0], vertex[1], vertex[2], p_intersection);
	return false;
}

bool Face3::is_degenerate() const {
	Vector3 normal = vec3_cross(vertex[0] - vertex[1], vertex[0] - vertex[2]);
	return (normal.length_squared() < (real_t)CMP_EPSILON2);
}

Face3::Side Face3::get_side_of(const Face3 &p_face, ClockDirection p_clock_dir) const {
	int over = 0, under = 0;

	Plane plane = get_plane(p_clock_dir);

	for (int i = 0; i < 3; i++) {
		const Vector3 &v = p_face.vertex[i];

		if (plane.has_point(v)) { //coplanar, don't bother
			continue;
		}

		if (plane.is_point_over(v)) {
			over++;
		} else {
			under++;
		}
	}

	if (over > 0 && under == 0) {
		return SIDE_OVER;
	} else if (under > 0 && over == 0) {
		return SIDE_UNDER;
	} else if (under == 0 && over == 0) {
		return SIDE_COPLANAR;
	} else {
		return SIDE_SPANNING;
	}
}

Vector3 Face3::get_random_point_inside() const {
	real_t a = Math::random(0, 1);
	real_t b = Math::random(0, 1);
	if (a > b) {
		SWAP(a, b);
	}

	return vertex[0] * a + vertex[1] * (b - a) + vertex[2] * (1.0 - b);
}

Plane Face3::get_plane(ClockDirection p_dir) const {
	return Plane(vertex[0], vertex[1], vertex[2], p_dir);
}

Vector3 Face3::get_median_point() const {
	return (vertex[0] + vertex[1] + vertex[2]) / 3.0;
}

real_t Face3::get_area() const {
	return vec3_cross(vertex[0] - vertex[1], vertex[0] - vertex[2]).length() * 0.5;
}

ClockDirection Face3::get_clock_dir() const {
	Vector3 normal = vec3_cross(vertex[0] - vertex[1], vertex[0] - vertex[2]);
	//printf("normal is %g,%g,%g x %g,%g,%g- wtfu is %g\n",tofloat(normal.x),tofloat(normal.y),tofloat(normal.z),tofloat(vertex[0].x),tofloat(vertex[0].y),tofloat(vertex[0].z),tofloat( normal.dot( vertex[0] ) ) );
	return (normal.dot(vertex[0]) >= 0) ? CLOCKWISE : COUNTERCLOCKWISE;
}

bool Face3::intersects_aabb(const AABB &p_aabb) const {
	/** TEST PLANE **/
	if (!p_aabb.intersects_plane(get_plane())) {
		return false;
	}

#define TEST_AXIS(m_ax)                                            \
	/** TEST FACE AXIS */                                          \
	{                                                              \
		real_t aabb_min = p_aabb.position.m_ax;                    \
		real_t aabb_max = p_aabb.position.m_ax + p_aabb.size.m_ax; \
		real_t tri_min = vertex[0].m_ax;                           \
		real_t tri_max = vertex[0].m_ax;                           \
		for (int i = 1; i < 3; i++) {                              \
			if (vertex[i].m_ax > tri_max)                          \
				tri_max = vertex[i].m_ax;                          \
			if (vertex[i].m_ax < tri_min)                          \
				tri_min = vertex[i].m_ax;                          \
		}                                                          \
                                                                   \
		if (tri_max < aabb_min || aabb_max < tri_min)              \
			return false;                                          \
	}

	TEST_AXIS(x);
	TEST_AXIS(y);
	TEST_AXIS(z);

	/** TEST ALL EDGES **/

	Vector3 edge_norms[3] = {
		vertex[0] - vertex[1],
		vertex[1] - vertex[2],
		vertex[2] - vertex[0],
	};

	for (int i = 0; i < 12; i++) {
		Vector3 from, to;
		p_aabb.get_edge(i, from, to);
		Vector3 e1 = from - to;
		for (int j = 0; j < 3; j++) {
			Vector3 e2 = edge_norms[j];

			Vector3 axis = vec3_cross(e1, e2);

			if (axis.length_squared() < 0.0001f) {
				continue; // coplanar
			}
			axis.normalize();

			real_t minA, maxA, minB, maxB;
			p_aabb.project_range_in_plane(Plane(axis, 0), minA, maxA);
			project_range(axis, Transform(), minB, maxB);

			if (maxA < minB || maxB < minA) {
				return false;
			}
		}
	}
	return true;
}

Face3::operator String() const {
	return String() + vertex[0] + ", " + vertex[1] + ", " + vertex[2];
}

void Face3::project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const {
	for (int i = 0; i < 3; i++) {
		Vector3 v = p_transform.xform(vertex[i]);
		real_t d = p_normal.dot(v);

		if (i == 0 || d > r_max) {
			r_max = d;
		}

		if (i == 0 || d < r_min) {
			r_min = d;
		}
	}
}

void Face3::get_support(const Vector3 &p_normal, const Transform &p_transform, Vector3 *p_vertices, int *p_count, int p_max) const {
#define _FACE_IS_VALID_SUPPORT_THRESHOLD 0.98
#define _EDGE_IS_VALID_SUPPORT_THRESHOLD 0.05

	if (p_max <= 0) {
		return;
	}

	Vector3 n = p_transform.basis.xform_inv(p_normal);

	/** TEST FACE AS SUPPORT **/
	if (get_plane().normal.dot(n) > (real_t)_FACE_IS_VALID_SUPPORT_THRESHOLD) {
		*p_count = MIN(3, p_max);

		for (int i = 0; i < *p_count; i++) {
			p_vertices[i] = p_transform.xform(vertex[i]);
		}

		return;
	}

	/** FIND SUPPORT VERTEX **/

	int vert_support_idx = -1;
	real_t support_max = 0;

	for (int i = 0; i < 3; i++) {
		real_t d = n.dot(vertex[i]);

		if (i == 0 || d > support_max) {
			support_max = d;
			vert_support_idx = i;
		}
	}

	/** TEST EDGES AS SUPPORT **/

	for (int i = 0; i < 3; i++) {
		if (i != vert_support_idx && i + 1 != vert_support_idx) {
			continue;
		}

		// check if edge is valid as a support
		real_t dot = (vertex[i] - vertex[(i + 1) % 3]).normalized().dot(n);
		dot = ABS(dot);
		if (dot < (real_t)_EDGE_IS_VALID_SUPPORT_THRESHOLD) {
			*p_count = MIN(2, p_max);

			for (int j = 0; j < *p_count; j++) {
				p_vertices[j] = p_transform.xform(vertex[(j + i) % 3]);
			}

			return;
		}
	}

	*p_count = 1;
	p_vertices[0] = p_transform.xform(vertex[vert_support_idx]);
}

Vector3 Face3::get_closest_point_to(const Vector3 &p_point) const {
	Vector3 edge0 = vertex[1] - vertex[0];
	Vector3 edge1 = vertex[2] - vertex[0];
	Vector3 v0 = vertex[0] - p_point;

	real_t a = edge0.dot(edge0);
	real_t b = edge0.dot(edge1);
	real_t c = edge1.dot(edge1);
	real_t d = edge0.dot(v0);
	real_t e = edge1.dot(v0);

	real_t det = a * c - b * b;
	real_t s = b * e - c * d;
	real_t t = b * d - a * e;

	if (s + t < det) {
		if (s < 0.f) {
			if (t < 0.f) {
				if (d < 0.f) {
					s = CLAMP(-d / a, 0.f, 1.f);
					t = 0.f;
				} else {
					s = 0.f;
					t = CLAMP(-e / c, 0.f, 1.f);
				}
			} else {
				s = 0.f;
				t = CLAMP(-e / c, 0.f, 1.f);
			}
		} else if (t < 0.f) {
			s = CLAMP(-d / a, 0.f, 1.f);
			t = 0.f;
		} else {
			real_t invDet = 1.f / det;
			s *= invDet;
			t *= invDet;
		}
	} else {
		if (s < 0.f) {
			real_t tmp0 = b + d;
			real_t tmp1 = c + e;
			if (tmp1 > tmp0) {
				real_t numer = tmp1 - tmp0;
				real_t denom = a - 2 * b + c;
				s = CLAMP(numer / denom, 0.f, 1.f);
				t = 1 - s;
			} else {
				t = CLAMP(-e / c, 0.f, 1.f);
				s = 0.f;
			}
		} else if (t < 0.f) {
			if (a + d > b + e) {
				real_t numer = c + e - b - d;
				real_t denom = a - 2 * b + c;
				s = CLAMP(numer / denom, 0.f, 1.f);
				t = 1 - s;
			} else {
				s = CLAMP(-d / a, 0.f, 1.f);
				t = 0.f;
			}
		} else {
			real_t numer = c + e - b - d;
			real_t denom = a - 2 * b + c;
			s = CLAMP(numer / denom, 0.f, 1.f);
			t = 1.f - s;
		}
	}

	return vertex[0] + s * edge0 + t * edge1;
}
#line 0

#line 1 "sfw/core/vector4i.cpp"
/*************************************************************************/
/*  vector4i.cpp                                                         */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

void Vector4i::set_axis(const int p_axis, const int32_t p_value) {
	ERR_FAIL_INDEX(p_axis, 4);
	coord[p_axis] = p_value;
}

int32_t Vector4i::get_axis(const int p_axis) const {
	ERR_FAIL_INDEX_V(p_axis, 4, 0);
	return operator[](p_axis);
}

Vector4i::Axis Vector4i::min_axis() const {
	uint32_t min_index = 0;
	int32_t min_value = x;
	for (uint32_t i = 1; i < 4; i++) {
		if (operator[](i) <= min_value) {
			min_index = i;
			min_value = operator[](i);
		}
	}
	return Vector4i::Axis(min_index);
}

Vector4i::Axis Vector4i::max_axis() const {
	uint32_t max_index = 0;
	int32_t max_value = x;
	for (uint32_t i = 1; i < 4; i++) {
		if (operator[](i) > max_value) {
			max_index = i;
			max_value = operator[](i);
		}
	}
	return Vector4i::Axis(max_index);
}

Vector4i Vector4i::clamp(const Vector4i &p_min, const Vector4i &p_max) const {
	return Vector4i(
			CLAMP(x, p_min.x, p_max.x),
			CLAMP(y, p_min.y, p_max.y),
			CLAMP(z, p_min.z, p_max.z),
			CLAMP(w, p_min.w, p_max.w));
}

Vector4i Vector4i::linear_interpolate(const Vector4i &p_to, const real_t p_weight) const {
	return Vector4i(
			x + (p_weight * (p_to.x - x)),
			y + (p_weight * (p_to.y - y)),
			z + (p_weight * (p_to.z - z)),
			w + (p_weight * (p_to.w - w)));
}

Vector4 Vector4i::to_vector4() const {
	return Vector4(x, y, z, w);
}

Vector4i::operator String() const {
	return "(" + itos(x) + ", " + itos(y) + ", " + itos(z) + ", " + itos(w) + ")";
}

Vector4i::operator Vector4() const {
	return Vector4(x, y, z, w);
}

/*
Vector4i::Vector4i(const Vector4 &p_vec4) {
	x = p_vec4.x;
	y = p_vec4.y;
	z = p_vec4.z;
	w = p_vec4.w;
}
*/
#line 0

#line 1 "sfw/core/transform.cpp"
/*************************************************************************/
/*  transform.cpp                                                        */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

void Transform::invert() {
	basis.transpose();
	origin = basis.xform(-origin);
}

Transform Transform::inverse() const {
	// FIXME: this function assumes the basis is a rotation matrix, with no scaling.
	// Transform::affine_inverse can handle matrices with scaling, so GDScript should eventually use that.
	Transform ret = *this;
	ret.invert();
	return ret;
}

void Transform::affine_invert() {
	basis.invert();
	origin = basis.xform(-origin);
}

Transform Transform::affine_inverse() const {
	Transform ret = *this;
	ret.affine_invert();
	return ret;
}

Transform Transform::rotated(const Vector3 &p_axis, real_t p_angle) const {
	// Equivalent to left multiplication
	Basis p_basis(p_axis, p_angle);
	return Transform(p_basis * basis, p_basis.xform(origin));
}

Transform Transform::rotated_local(const Vector3 &p_axis, real_t p_angle) const {
	// Equivalent to right multiplication
	Basis p_basis(p_axis, p_angle);
	return Transform(basis * p_basis, origin);
}

void Transform::rotate(const Vector3 &p_axis, real_t p_phi) {
	*this = rotated(p_axis, p_phi);
}

void Transform::rotate_local(const Vector3 &p_axis, real_t p_phi) {
	*this = rotated_local(p_axis, p_phi);
}

void Transform::rotate_basis(const Vector3 &p_axis, real_t p_phi) {
	basis.rotate(p_axis, p_phi);
}

void Transform::set_look_at(const Vector3 &p_eye, const Vector3 &p_target, const Vector3 &p_up) {
#ifdef MATH_CHECKS
	ERR_FAIL_COND(p_eye == p_target);
	ERR_FAIL_COND(p_up.length() == 0);
#endif
	// Reference: MESA source code
	Vector3 v_x, v_y, v_z;

	/* Make rotation matrix */

	/* Z vector */
	v_z = p_eye - p_target;

	v_z.normalize();

	v_y = p_up;

	v_x = v_y.cross(v_z);
#ifdef MATH_CHECKS
	ERR_FAIL_COND(v_x.length() == 0);
#endif

	/* Recompute Y = Z cross X */
	v_y = v_z.cross(v_x);

	v_x.normalize();
	v_y.normalize();

	basis.set(v_x, v_y, v_z);

	origin = p_eye;
}

Transform Transform::looking_at(const Vector3 &p_target, const Vector3 &p_up) const {
	Transform t = *this;
	t.set_look_at(origin, p_target, p_up);
	return t;
}

void Transform::scale(const Vector3 &p_scale) {
	basis.scale(p_scale);
	origin *= p_scale;
}

Transform Transform::scaled(const Vector3 &p_scale) const {
	// Equivalent to left multiplication
	return Transform(basis.scaled(p_scale), origin * p_scale);
}

Transform Transform::scaled_local(const Vector3 &p_scale) const {
	// Equivalent to right multiplication
	return Transform(basis.scaled_local(p_scale), origin);
}

void Transform::scale_basis(const Vector3 &p_scale) {
	basis.scale(p_scale);
}

void Transform::translate_local(real_t p_tx, real_t p_ty, real_t p_tz) {
	translate_local(Vector3(p_tx, p_ty, p_tz));
}
void Transform::translate_local(const Vector3 &p_translation) {
	for (int i = 0; i < 3; i++) {
		origin[i] += basis[i].dot(p_translation);
	}
}

void Transform::translate_localr(real_t p_tx, real_t p_ty, real_t p_tz) {
	translate_local(Vector3(p_tx, p_ty, p_tz));
}
void Transform::translate_localv(const Vector3 &p_translation) {
	for (int i = 0; i < 3; i++) {
		origin[i] += basis[i].dot(p_translation);
	}
}

Transform Transform::translated(const Vector3 &p_translation) const {
	// Equivalent to left multiplication
	return Transform(basis, origin + p_translation);
}

Transform Transform::translated_local(const Vector3 &p_translation) const {
	// Equivalent to right multiplication
	return Transform(basis, origin + basis.xform(p_translation));
}

void Transform::orthonormalize() {
	basis.orthonormalize();
}

Transform Transform::orthonormalized() const {
	Transform _copy = *this;
	_copy.orthonormalize();
	return _copy;
}

void Transform::orthogonalize() {
	basis.orthogonalize();
}

Transform Transform::orthogonalized() const {
	Transform _copy = *this;
	_copy.orthogonalize();
	return _copy;
}

bool Transform::is_equal_approx(const Transform &p_transform) const {
	return basis.is_equal_approx(p_transform.basis) && origin.is_equal_approx(p_transform.origin);
}

bool Transform::operator==(const Transform &p_transform) const {
	return (basis == p_transform.basis && origin == p_transform.origin);
}
bool Transform::operator!=(const Transform &p_transform) const {
	return (basis != p_transform.basis || origin != p_transform.origin);
}

void Transform::operator*=(const Transform &p_transform) {
	origin = xform(p_transform.origin);
	basis *= p_transform.basis;
}

Transform Transform::operator*(const Transform &p_transform) const {
	Transform t = *this;
	t *= p_transform;
	return t;
}

void Transform::operator*=(const real_t p_val) {
	origin *= p_val;
	basis *= p_val;
}

Transform Transform::operator*(const real_t p_val) const {
	Transform ret(*this);
	ret *= p_val;
	return ret;
}

Transform Transform::spherical_interpolate_with(const Transform &p_transform, real_t p_c) const {
	/* not sure if very "efficient" but good enough? */

	Transform interp;

	Vector3 src_scale = basis.get_scale();
	Quaternion src_rot = basis.get_rotation_quaternion();
	Vector3 src_loc = origin;

	Vector3 dst_scale = p_transform.basis.get_scale();
	Quaternion dst_rot = p_transform.basis.get_rotation_quaternion();
	Vector3 dst_loc = p_transform.origin;

	interp.basis.set_quaternion_scale(src_rot.slerp(dst_rot, p_c).normalized(), src_scale.linear_interpolate(dst_scale, p_c));
	interp.origin = src_loc.linear_interpolate(dst_loc, p_c);

	return interp;
}

Transform Transform::interpolate_with(const Transform &p_transform, real_t p_c) const {
	/* not sure if very "efficient" but good enough? */

	Vector3 src_scale = basis.get_scale();
	Quaternion src_rot = basis.get_rotation_quaternion();
	Vector3 src_loc = origin;

	Vector3 dst_scale = p_transform.basis.get_scale();
	Quaternion dst_rot = p_transform.basis.get_rotation_quaternion();
	Vector3 dst_loc = p_transform.origin;

	Transform interp;
	interp.basis.set_quaternion_scale(src_rot.slerp(dst_rot, p_c).normalized(), src_scale.linear_interpolate(dst_scale, p_c));
	interp.origin = src_loc.linear_interpolate(dst_loc, p_c);

	return interp;
}

Transform::operator String() const {
	return "[X: " + basis.get_axis(0).operator String() +
			", Y: " + basis.get_axis(1).operator String() +
			", Z: " + basis.get_axis(2).operator String() +
			", O: " + origin.operator String() + "]";
}

Transform::Transform(const Basis &p_basis, const Vector3 &p_origin) :
		basis(p_basis),
		origin(p_origin) {
}

Transform::Transform(real_t xx, real_t xy, real_t xz, real_t yx, real_t yy, real_t yz, real_t zx, real_t zy, real_t zz, real_t ox, real_t oy, real_t oz) {
	basis = Basis(xx, xy, xz, yx, yy, yz, zx, zy, zz);
	origin = Vector3(ox, oy, oz);
}

Transform::Transform(const Vector3 &p_x, const Vector3 &p_y, const Vector3 &p_z, const Vector3 &p_origin) :
		origin(p_origin) {
	basis.set_column(0, p_x);
	basis.set_column(1, p_y);
	basis.set_column(2, p_z);
}
#line 0

#line 1 "sfw/core/color.cpp"
/*************************************************************************/
/*  color.cpp                                                            */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

uint32_t Color::to_argb32() const {
	uint32_t c = (uint8_t)Math::round(a * 255);
	c <<= 8;
	c |= (uint8_t)Math::round(r * 255);
	c <<= 8;
	c |= (uint8_t)Math::round(g * 255);
	c <<= 8;
	c |= (uint8_t)Math::round(b * 255);

	return c;
}

uint32_t Color::to_abgr32() const {
	uint32_t c = (uint8_t)Math::round(a * 255);
	c <<= 8;
	c |= (uint8_t)Math::round(b * 255);
	c <<= 8;
	c |= (uint8_t)Math::round(g * 255);
	c <<= 8;
	c |= (uint8_t)Math::round(r * 255);

	return c;
}

uint32_t Color::to_rgba32() const {
	uint32_t c = (uint8_t)Math::round(r * 255);
	c <<= 8;
	c |= (uint8_t)Math::round(g * 255);
	c <<= 8;
	c |= (uint8_t)Math::round(b * 255);
	c <<= 8;
	c |= (uint8_t)Math::round(a * 255);

	return c;
}

uint64_t Color::to_abgr64() const {
	uint64_t c = (uint16_t)Math::round(a * 65535);
	c <<= 16;
	c |= (uint16_t)Math::round(b * 65535);
	c <<= 16;
	c |= (uint16_t)Math::round(g * 65535);
	c <<= 16;
	c |= (uint16_t)Math::round(r * 65535);

	return c;
}

uint64_t Color::to_argb64() const {
	uint64_t c = (uint16_t)Math::round(a * 65535);
	c <<= 16;
	c |= (uint16_t)Math::round(r * 65535);
	c <<= 16;
	c |= (uint16_t)Math::round(g * 65535);
	c <<= 16;
	c |= (uint16_t)Math::round(b * 65535);

	return c;
}

uint64_t Color::to_rgba64() const {
	uint64_t c = (uint16_t)Math::round(r * 65535);
	c <<= 16;
	c |= (uint16_t)Math::round(g * 65535);
	c <<= 16;
	c |= (uint16_t)Math::round(b * 65535);
	c <<= 16;
	c |= (uint16_t)Math::round(a * 65535);

	return c;
}

float Color::get_h() const {
	float min = MIN(r, g);
	min = MIN(min, b);
	float max = MAX(r, g);
	max = MAX(max, b);

	float delta = max - min;

	if (delta == 0) {
		return 0;
	}

	float h;
	if (r == max) {
		h = (g - b) / delta; // between yellow & magenta
	} else if (g == max) {
		h = 2 + (b - r) / delta; // between cyan & yellow
	} else {
		h = 4 + (r - g) / delta; // between magenta & cyan
	}

	h /= 6.0;
	if (h < 0) {
		h += 1.0;
	}

	return h;
}

float Color::get_s() const {
	float min = MIN(r, g);
	min = MIN(min, b);
	float max = MAX(r, g);
	max = MAX(max, b);

	float delta = max - min;

	return (max != 0) ? (delta / max) : 0;
}

float Color::get_v() const {
	float max = MAX(r, g);
	max = MAX(max, b);
	return max;
}

void Color::set_hsv(float p_h, float p_s, float p_v, float p_alpha) {
	int i;
	float f, p, q, t;
	a = p_alpha;

	if (p_s == 0) {
		// acp_hromatic (grey)
		r = g = b = p_v;
		return;
	}

	p_h *= 6.0;
	p_h = Math::fmod(p_h, 6);
	i = Math::floor(p_h);

	f = p_h - i;
	p = p_v * (1 - p_s);
	q = p_v * (1 - p_s * f);
	t = p_v * (1 - p_s * (1 - f));

	switch (i) {
		case 0: // Red is the dominant color
			r = p_v;
			g = t;
			b = p;
			break;
		case 1: // Green is the dominant color
			r = q;
			g = p_v;
			b = p;
			break;
		case 2:
			r = p;
			g = p_v;
			b = t;
			break;
		case 3: // Blue is the dominant color
			r = p;
			g = q;
			b = p_v;
			break;
		case 4:
			r = t;
			g = p;
			b = p_v;
			break;
		default: // (5) Red is the dominant color
			r = p_v;
			g = p;
			b = q;
			break;
	}
}

bool Color::is_equal_approx(const Color &p_color) const {
	return Math::is_equal_approx(r, p_color.r) && Math::is_equal_approx(g, p_color.g) && Math::is_equal_approx(b, p_color.b) && Math::is_equal_approx(a, p_color.a);
}

Color Color::clamp(const Color &p_min, const Color &p_max) const {
	return Color(
			CLAMP(r, p_min.r, p_max.r),
			CLAMP(g, p_min.g, p_max.g),
			CLAMP(b, p_min.b, p_max.b),
			CLAMP(a, p_min.a, p_max.a));
}

void Color::invert() {
	r = 1.0 - r;
	g = 1.0 - g;
	b = 1.0 - b;
}
void Color::contrast() {
	r = Math::fmod(r + 0.5, 1.0);
	g = Math::fmod(g + 0.5, 1.0);
	b = Math::fmod(b + 0.5, 1.0);
}

Color Color::hex(uint32_t p_hex) {
	float a = (p_hex & 0xFF) / 255.0;
	p_hex >>= 8;
	float b = (p_hex & 0xFF) / 255.0;
	p_hex >>= 8;
	float g = (p_hex & 0xFF) / 255.0;
	p_hex >>= 8;
	float r = (p_hex & 0xFF) / 255.0;

	return Color(r, g, b, a);
}

Color Color::hex64(uint64_t p_hex) {
	float a = (p_hex & 0xFFFF) / 65535.0;
	p_hex >>= 16;
	float b = (p_hex & 0xFFFF) / 65535.0;
	p_hex >>= 16;
	float g = (p_hex & 0xFFFF) / 65535.0;
	p_hex >>= 16;
	float r = (p_hex & 0xFFFF) / 65535.0;

	return Color(r, g, b, a);
}

Color Color::from_rgbe9995(uint32_t p_rgbe) {
	float r = p_rgbe & 0x1ff;
	float g = (p_rgbe >> 9) & 0x1ff;
	float b = (p_rgbe >> 18) & 0x1ff;
	float e = (p_rgbe >> 27);
	float m = Math::pow(2, e - 15.0 - 9.0);

	float rd = r * m;
	float gd = g * m;
	float bd = b * m;

	return Color(rd, gd, bd, 1.0f);
}

static float _parse_col(const String &p_str, int p_ofs) {
	int ig = 0;

	for (int i = 0; i < 2; i++) {
		int c = p_str[i + p_ofs];
		int v = 0;

		if (c >= '0' && c <= '9') {
			v = c - '0';
		} else if (c >= 'a' && c <= 'f') {
			v = c - 'a';
			v += 10;
		} else if (c >= 'A' && c <= 'F') {
			v = c - 'A';
			v += 10;
		} else {
			return -1;
		}

		if (i == 0) {
			ig += v * 16;
		} else {
			ig += v;
		}
	}

	return ig;
}

Color Color::inverted() const {
	Color c = *this;
	c.invert();
	return c;
}

Color Color::contrasted() const {
	Color c = *this;
	c.contrast();
	return c;
}

Color Color::html(const String &p_color) {
	String color = p_color;
	if (color.length() == 0) {
		return Color();
	}
	if (color[0] == '#') {
		color = color.substr(1, color.length() - 1);
	}
	if (color.length() == 3 || color.length() == 4) {
		String exp_color;
		for (int i = 0; i < color.length(); i++) {
			exp_color += color[i];
			exp_color += color[i];
		}
		color = exp_color;
	}

	bool alpha = false;

	if (color.length() == 8) {
		alpha = true;
	} else if (color.length() == 6) {
		alpha = false;
	} else {
		ERR_FAIL_V_MSG(Color(), "Invalid color code: " + p_color + ".");
	}

	int a = 255;
	if (alpha) {
		a = _parse_col(color, 0);
		ERR_FAIL_COND_V_MSG(a < 0, Color(), "Invalid color code: " + p_color + ".");
	}

	int from = alpha ? 2 : 0;

	int r = _parse_col(color, from + 0);
	ERR_FAIL_COND_V_MSG(r < 0, Color(), "Invalid color code: " + p_color + ".");
	int g = _parse_col(color, from + 2);
	ERR_FAIL_COND_V_MSG(g < 0, Color(), "Invalid color code: " + p_color + ".");
	int b = _parse_col(color, from + 4);
	ERR_FAIL_COND_V_MSG(b < 0, Color(), "Invalid color code: " + p_color + ".");

	return Color(r / 255.0, g / 255.0, b / 255.0, a / 255.0);
}

bool Color::html_is_valid(const String &p_color) {
	String color = p_color;

	if (color.length() == 0) {
		return false;
	}
	if (color[0] == '#') {
		color = color.substr(1, color.length() - 1);
	}

	bool alpha = false;

	if (color.length() == 8) {
		alpha = true;
	} else if (color.length() == 6) {
		alpha = false;
	} else {
		return false;
	}

	if (alpha) {
		int a = _parse_col(color, 0);
		if (a < 0) {
			return false;
		}
	}

	int from = alpha ? 2 : 0;

	int r = _parse_col(color, from + 0);
	if (r < 0) {
		return false;
	}
	int g = _parse_col(color, from + 2);
	if (g < 0) {
		return false;
	}
	int b = _parse_col(color, from + 4);
	if (b < 0) {
		return false;
	}

	return true;
}

String _to_hex(float p_val) {
	int v = Math::round(p_val * 255);
	v = CLAMP(v, 0, 255);
	String ret;

	for (int i = 0; i < 2; i++) {
		CharType c[2] = { 0, 0 };
		int lv = v & 0xF;
		if (lv < 10) {
			c[0] = '0' + lv;
		} else {
			c[0] = 'a' + lv - 10;
		}

		v >>= 4;
		String cs = (const CharType *)c;
		ret = cs + ret;
	}

	return ret;
}

String Color::to_html(bool p_alpha) const {
	String txt;
	txt += _to_hex(r);
	txt += _to_hex(g);
	txt += _to_hex(b);
	if (p_alpha) {
		txt = _to_hex(a) + txt;
	}
	return txt;
}

Color Color::from_hsv(float p_h, float p_s, float p_v, float p_a) const {
	Color c;
	c.set_hsv(p_h, p_s, p_v, p_a);
	return c;
}

Color::operator String() const {
	return "(" + String::num(r, 4) + ", " + String::num(g, 4) + ", " + String::num(b, 4) + ", " + String::num(a, 4) + ")";
}

Color Color::operator+(const Color &p_color) const {
	return Color(
			r + p_color.r,
			g + p_color.g,
			b + p_color.b,
			a + p_color.a);
}

void Color::operator+=(const Color &p_color) {
	r = r + p_color.r;
	g = g + p_color.g;
	b = b + p_color.b;
	a = a + p_color.a;
}

Color Color::operator-(const Color &p_color) const {
	return Color(
			r - p_color.r,
			g - p_color.g,
			b - p_color.b,
			a - p_color.a);
}

void Color::operator-=(const Color &p_color) {
	r = r - p_color.r;
	g = g - p_color.g;
	b = b - p_color.b;
	a = a - p_color.a;
}

Color Color::operator*(const Color &p_color) const {
	return Color(
			r * p_color.r,
			g * p_color.g,
			b * p_color.b,
			a * p_color.a);
}

Color Color::operator*(const real_t &rvalue) const {
	return Color(
			r * rvalue,
			g * rvalue,
			b * rvalue,
			a * rvalue);
}

void Color::operator*=(const Color &p_color) {
	r = r * p_color.r;
	g = g * p_color.g;
	b = b * p_color.b;
	a = a * p_color.a;
}

void Color::operator*=(const real_t &rvalue) {
	r = r * rvalue;
	g = g * rvalue;
	b = b * rvalue;
	a = a * rvalue;
}

Color Color::operator/(const Color &p_color) const {
	return Color(
			r / p_color.r,
			g / p_color.g,
			b / p_color.b,
			a / p_color.a);
}

Color Color::operator/(const real_t &rvalue) const {
	return Color(
			r / rvalue,
			g / rvalue,
			b / rvalue,
			a / rvalue);
}

void Color::operator/=(const Color &p_color) {
	r = r / p_color.r;
	g = g / p_color.g;
	b = b / p_color.b;
	a = a / p_color.a;
}

void Color::operator/=(const real_t &rvalue) {
	if (rvalue == 0) {
		r = 1.0;
		g = 1.0;
		b = 1.0;
		a = 1.0;
	} else {
		r = r / rvalue;
		g = g / rvalue;
		b = b / rvalue;
		a = a / rvalue;
	}
};

Color Color::operator-() const {
	return Color(
			1.0 - r,
			1.0 - g,
			1.0 - b,
			1.0 - a);
}
#line 0

#line 1 "sfw/core/quaternion.cpp"
/*************************************************************************/
/*  quaternion.cpp                                                       */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

real_t Quaternion::angle_to(const Quaternion &p_to) const {
	real_t d = dot(p_to);

	// acos does clamping.
	return Math::acos(d * d * 2 - 1);
}

// set_euler_xyz expects a vector containing the Euler angles in the format
// (ax,ay,az), where ax is the angle of rotation around x axis,
// and similar for other axes.
// This implementation uses XYZ convention (Z is the first rotation).
void Quaternion::set_euler_xyz(const Vector3 &p_euler) {
	real_t half_a1 = p_euler.x * 0.5f;
	real_t half_a2 = p_euler.y * 0.5f;
	real_t half_a3 = p_euler.z * 0.5f;

	// R = X(a1).Y(a2).Z(a3) convention for Euler angles.
	// Conversion to quaternion as listed in https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770024290.pdf (page A-2)
	// a3 is the angle of the first rotation, following the notation in this reference.

	real_t cos_a1 = Math::cos(half_a1);
	real_t sin_a1 = Math::sin(half_a1);
	real_t cos_a2 = Math::cos(half_a2);
	real_t sin_a2 = Math::sin(half_a2);
	real_t cos_a3 = Math::cos(half_a3);
	real_t sin_a3 = Math::sin(half_a3);

	set(sin_a1 * cos_a2 * cos_a3 + sin_a2 * sin_a3 * cos_a1,
			-sin_a1 * sin_a3 * cos_a2 + sin_a2 * cos_a1 * cos_a3,
			sin_a1 * sin_a2 * cos_a3 + sin_a3 * cos_a1 * cos_a2,
			-sin_a1 * sin_a2 * sin_a3 + cos_a1 * cos_a2 * cos_a3);
}

// get_euler_xyz returns a vector containing the Euler angles in the format
// (ax,ay,az), where ax is the angle of rotation around x axis,
// and similar for other axes.
// This implementation uses XYZ convention (Z is the first rotation).
Vector3 Quaternion::get_euler_xyz() const {
	Basis m(*this);
	return m.get_euler_xyz();
}

// set_euler_yxz expects a vector containing the Euler angles in the format
// (ax,ay,az), where ax is the angle of rotation around x axis,
// and similar for other axes.
// This implementation uses YXZ convention (Z is the first rotation).
void Quaternion::set_euler_yxz(const Vector3 &p_euler) {
	real_t half_a1 = p_euler.y * 0.5f;
	real_t half_a2 = p_euler.x * 0.5f;
	real_t half_a3 = p_euler.z * 0.5f;

	// R = Y(a1).X(a2).Z(a3) convention for Euler angles.
	// Conversion to quaternion as listed in https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770024290.pdf (page A-6)
	// a3 is the angle of the first rotation, following the notation in this reference.

	real_t cos_a1 = Math::cos(half_a1);
	real_t sin_a1 = Math::sin(half_a1);
	real_t cos_a2 = Math::cos(half_a2);
	real_t sin_a2 = Math::sin(half_a2);
	real_t cos_a3 = Math::cos(half_a3);
	real_t sin_a3 = Math::sin(half_a3);

	set(sin_a1 * cos_a2 * sin_a3 + cos_a1 * sin_a2 * cos_a3,
			sin_a1 * cos_a2 * cos_a3 - cos_a1 * sin_a2 * sin_a3,
			-sin_a1 * sin_a2 * cos_a3 + cos_a1 * cos_a2 * sin_a3,
			sin_a1 * sin_a2 * sin_a3 + cos_a1 * cos_a2 * cos_a3);
}

// get_euler_yxz returns a vector containing the Euler angles in the format
// (ax,ay,az), where ax is the angle of rotation around x axis,
// and similar for other axes.
// This implementation uses YXZ convention (Z is the first rotation).
Vector3 Quaternion::get_euler_yxz() const {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_V_MSG(!is_normalized(), Vector3(0, 0, 0), "The quaternion must be normalized.");
#endif
	Basis m(*this);
	return m.get_euler_yxz();
}

void Quaternion::operator*=(const Quaternion &p_q) {
	set(w * p_q.x + x * p_q.w + y * p_q.z - z * p_q.y,
			w * p_q.y + y * p_q.w + z * p_q.x - x * p_q.z,
			w * p_q.z + z * p_q.w + x * p_q.y - y * p_q.x,
			w * p_q.w - x * p_q.x - y * p_q.y - z * p_q.z);
}

Quaternion Quaternion::operator*(const Quaternion &p_q) const {
	Quaternion r = *this;
	r *= p_q;
	return r;
}

bool Quaternion::is_equal_approx(const Quaternion &p_quat) const {
	return Math::is_equal_approx(x, p_quat.x) && Math::is_equal_approx(y, p_quat.y) && Math::is_equal_approx(z, p_quat.z) && Math::is_equal_approx(w, p_quat.w);
}

real_t Quaternion::length() const {
	return Math::sqrt(length_squared());
}

void Quaternion::normalize() {
	*this /= length();
}

Quaternion Quaternion::normalized() const {
	return *this / length();
}

bool Quaternion::is_normalized() const {
	return Math::is_equal_approx(length_squared(), 1, (real_t)UNIT_EPSILON); //use less epsilon
}

Quaternion Quaternion::inverse() const {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_V_MSG(!is_normalized(), Quaternion(), "The quaternion must be normalized.");
#endif
	return Quaternion(-x, -y, -z, w);
}

Quaternion Quaternion::log() const {
	Quaternion src = *this;
	Vector3 src_v = src.get_axis() * src.get_angle();
	return Quaternion(src_v.x, src_v.y, src_v.z, 0);
}

Quaternion Quaternion::exp() const {
	Quaternion src = *this;
	Vector3 src_v = Vector3(src.x, src.y, src.z);
	float theta = src_v.length();
	if (theta < CMP_EPSILON) {
		return Quaternion(0, 0, 0, 1);
	}
	return Quaternion(src_v.normalized(), theta);
}

Quaternion Quaternion::slerp(const Quaternion &p_to, const real_t &p_weight) const {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_V_MSG(!is_normalized(), Quaternion(), "The start quaternion must be normalized.");
	ERR_FAIL_COND_V_MSG(!p_to.is_normalized(), Quaternion(), "The end quaternion must be normalized.");
#endif
	Quaternion to1;
	real_t omega, cosom, sinom, scale0, scale1;

	// calc cosine
	cosom = dot(p_to);

	// adjust signs (if necessary)
	if (cosom < 0) {
		cosom = -cosom;
		to1.x = -p_to.x;
		to1.y = -p_to.y;
		to1.z = -p_to.z;
		to1.w = -p_to.w;
	} else {
		to1.x = p_to.x;
		to1.y = p_to.y;
		to1.z = p_to.z;
		to1.w = p_to.w;
	}

	// calculate coefficients

	if ((1 - cosom) > (real_t)CMP_EPSILON) {
		// standard case (slerp)
		omega = Math::acos(cosom);
		sinom = Math::sin(omega);
		scale0 = Math::sin((1 - p_weight) * omega) / sinom;
		scale1 = Math::sin(p_weight * omega) / sinom;
	} else {
		// "from" and "to" quaternions are very close
		//  ... so we can do a linear interpolation
		scale0 = 1 - p_weight;
		scale1 = p_weight;
	}
	// calculate final values
	return Quaternion(
			scale0 * x + scale1 * to1.x,
			scale0 * y + scale1 * to1.y,
			scale0 * z + scale1 * to1.z,
			scale0 * w + scale1 * to1.w);
}

Quaternion Quaternion::slerpni(const Quaternion &p_to, const real_t &p_weight) const {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_V_MSG(!is_normalized(), Quaternion(), "The start quaternion must be normalized.");
	ERR_FAIL_COND_V_MSG(!p_to.is_normalized(), Quaternion(), "The end quaternion must be normalized.");
#endif
	const Quaternion &from = *this;

	real_t dot = from.dot(p_to);

	if (Math::absf(dot) > 0.9999f) {
		return from;
	}

	real_t theta = Math::acos(dot),
		   sinT = 1 / Math::sin(theta),
		   newFactor = Math::sin(p_weight * theta) * sinT,
		   invFactor = Math::sin((1 - p_weight) * theta) * sinT;

	return Quaternion(invFactor * from.x + newFactor * p_to.x,
			invFactor * from.y + newFactor * p_to.y,
			invFactor * from.z + newFactor * p_to.z,
			invFactor * from.w + newFactor * p_to.w);
}

Quaternion Quaternion::cubic_slerp(const Quaternion &p_b, const Quaternion &p_pre_a, const Quaternion &p_post_b, const real_t &p_weight) const {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_V_MSG(!is_normalized(), Quaternion(), "The start quaternion must be normalized.");
	ERR_FAIL_COND_V_MSG(!p_b.is_normalized(), Quaternion(), "The end quaternion must be normalized.");
#endif
	//the only way to do slerp :|
	real_t t2 = (1 - p_weight) * p_weight * 2;
	Quaternion sp = this->slerp(p_b, p_weight);
	Quaternion sq = p_pre_a.slerpni(p_post_b, p_weight);
	return sp.slerpni(sq, t2);
}

Quaternion Quaternion::spherical_cubic_interpolate(const Quaternion &p_b, const Quaternion &p_pre_a, const Quaternion &p_post_b, const real_t &p_weight) const {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_V_MSG(!is_normalized(), Quaternion(), "The start quaternion must be normalized.");
	ERR_FAIL_COND_V_MSG(!p_b.is_normalized(), Quaternion(), "The end quaternion must be normalized.");
#endif
	Quaternion from_q = *this;
	Quaternion pre_q = p_pre_a;
	Quaternion to_q = p_b;
	Quaternion post_q = p_post_b;

	// Align flip phases.
	from_q = Basis(from_q).get_rotation_quaternion();
	pre_q = Basis(pre_q).get_rotation_quaternion();
	to_q = Basis(to_q).get_rotation_quaternion();
	post_q = Basis(post_q).get_rotation_quaternion();

	// Flip quaternions to shortest path if necessary.
	bool flip1 = signbit(from_q.dot(pre_q));
	pre_q = flip1 ? -pre_q : pre_q;
	bool flip2 = signbit(from_q.dot(to_q));
	to_q = flip2 ? -to_q : to_q;
	bool flip3 = flip2 ? to_q.dot(post_q) <= 0 : signbit(to_q.dot(post_q));
	post_q = flip3 ? -post_q : post_q;

	// Calc by Expmap in from_q space.
	Quaternion ln_from = Quaternion(0, 0, 0, 0);
	Quaternion ln_to = (from_q.inverse() * to_q).log();
	Quaternion ln_pre = (from_q.inverse() * pre_q).log();
	Quaternion ln_post = (from_q.inverse() * post_q).log();
	Quaternion ln = Quaternion(0, 0, 0, 0);
	ln.x = Math::cubic_interpolate(ln_from.x, ln_to.x, ln_pre.x, ln_post.x, p_weight);
	ln.y = Math::cubic_interpolate(ln_from.y, ln_to.y, ln_pre.y, ln_post.y, p_weight);
	ln.z = Math::cubic_interpolate(ln_from.z, ln_to.z, ln_pre.z, ln_post.z, p_weight);
	Quaternion q1 = from_q * ln.exp();

	// Calc by Expmap in to_q space.
	ln_from = (to_q.inverse() * from_q).log();
	ln_to = Quaternion(0, 0, 0, 0);
	ln_pre = (to_q.inverse() * pre_q).log();
	ln_post = (to_q.inverse() * post_q).log();
	ln = Quaternion(0, 0, 0, 0);
	ln.x = Math::cubic_interpolate(ln_from.x, ln_to.x, ln_pre.x, ln_post.x, p_weight);
	ln.y = Math::cubic_interpolate(ln_from.y, ln_to.y, ln_pre.y, ln_post.y, p_weight);
	ln.z = Math::cubic_interpolate(ln_from.z, ln_to.z, ln_pre.z, ln_post.z, p_weight);
	Quaternion q2 = to_q * ln.exp();

	// To cancel error made by Expmap ambiguity, do blends.
	return q1.slerp(q2, p_weight);
}

Vector3 Quaternion::get_axis() const {
	if (Math::abs(w) > 1 - CMP_EPSILON) {
		return Vector3(x, y, z);
	}
	real_t r = ((real_t)1) / Math::sqrt(1 - w * w);
	return Vector3(x * r, y * r, z * r);
}

float Quaternion::get_angle() const {
	return 2 * Math::acos(w);
}

Quaternion::operator String() const {
	return "(" + String::num_real(x) + ", " + String::num_real(y) + ", " + String::num_real(z) + ", " + String::num_real(w) + ")";
}

void Quaternion::set_axis_angle(const Vector3 &axis, const real_t &angle) {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_MSG(!axis.is_normalized(), "The axis Vector3 must be normalized.");
#endif
	real_t d = axis.length();
	if (d == 0) {
		set(0, 0, 0, 0);
	} else {
		real_t sin_angle = Math::sin(angle * 0.5f);
		real_t cos_angle = Math::cos(angle * 0.5f);
		real_t s = sin_angle / d;
		set(axis.x * s, axis.y * s, axis.z * s,
				cos_angle);
	}
}
#line 0

#line 1 "sfw/core/plane.cpp"
/*************************************************************************/
/*  plane.cpp                                                            */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

void Plane::set_normal(const Vector3 &p_normal) {
	normal = p_normal;
}

void Plane::normalize() {
	real_t l = normal.length();
	if (l == 0) {
		*this = Plane(0, 0, 0, 0);
		return;
	}
	normal /= l;
	d /= l;
}

Plane Plane::normalized() const {
	Plane p = *this;
	p.normalize();
	return p;
}

Vector3 Plane::get_any_point() const {
	return get_normal() * d;
}

Vector3 Plane::get_any_perpendicular_normal() const {
	static const Vector3 p1 = Vector3(1, 0, 0);
	static const Vector3 p2 = Vector3(0, 1, 0);
	Vector3 p;

	if (ABS(normal.dot(p1)) > 0.99f) { // if too similar to p1
		p = p2; // use p2
	} else {
		p = p1; // use p1
	}

	p -= normal * normal.dot(p);
	p.normalize();

	return p;
}

/* intersections */

bool Plane::intersect_3(const Plane &p_plane1, const Plane &p_plane2, Vector3 *r_result) const {
	const Plane &p_plane0 = *this;
	Vector3 normal0 = p_plane0.normal;
	Vector3 normal1 = p_plane1.normal;
	Vector3 normal2 = p_plane2.normal;

	real_t denom = vec3_cross(normal0, normal1).dot(normal2);

	if (Math::is_zero_approx(denom)) {
		return false;
	}

	if (r_result) {
		*r_result = ((vec3_cross(normal1, normal2) * p_plane0.d) +
							(vec3_cross(normal2, normal0) * p_plane1.d) +
							(vec3_cross(normal0, normal1) * p_plane2.d)) /
				denom;
	}

	return true;
}

bool Plane::intersects_ray(const Vector3 &p_from, const Vector3 &p_dir, Vector3 *p_intersection) const {
	Vector3 segment = p_dir;
	real_t den = normal.dot(segment);

	//printf("den is %i\n",den);
	if (Math::is_zero_approx(den)) {
		return false;
	}

	real_t dist = (normal.dot(p_from) - d) / den;
	//printf("dist is %i\n",dist);

	if (dist > (real_t)CMP_EPSILON) { //this is a ray, before the emitting pos (p_from) doesn't exist

		return false;
	}

	dist = -dist;
	*p_intersection = p_from + segment * dist;

	return true;
}

bool Plane::intersects_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 *p_intersection) const {
	Vector3 segment = p_begin - p_end;
	real_t den = normal.dot(segment);

	//printf("den is %i\n",den);
	if (Math::is_zero_approx(den)) {
		return false;
	}

	real_t dist = (normal.dot(p_begin) - d) / den;
	//printf("dist is %i\n",dist);

	if (dist < (real_t)-CMP_EPSILON || dist > (1 + (real_t)CMP_EPSILON)) {
		return false;
	}

	dist = -dist;
	*p_intersection = p_begin + segment * dist;

	return true;
}

/* misc */

bool Plane::is_equal_approx(const Plane &p_plane) const {
	return normal.is_equal_approx(p_plane.normal) && Math::is_equal_approx(d, p_plane.d);
}

bool Plane::is_equal_approx_any_side(const Plane &p_plane) const {
	return (normal.is_equal_approx(p_plane.normal) && Math::is_equal_approx(d, p_plane.d)) || (normal.is_equal_approx(-p_plane.normal) && Math::is_equal_approx(d, -p_plane.d));
}

Plane::operator String() const {
	return "[N: " + normal.operator String() + ", D: " + String::num_real(d) + "]";
}
#line 0

#line 1 "sfw/core/vector2i.cpp"
/*************************************************************************/
/*  vector2i.cpp                                                         */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

Vector2i Vector2i::clamp(const Vector2i &p_min, const Vector2i &p_max) const {
	return Vector2i(
			CLAMP(x, p_min.x, p_max.x),
			CLAMP(y, p_min.y, p_max.y));
}

int64_t Vector2i::length_squared() const {
	return x * (int64_t)x + y * (int64_t)y;
}

double Vector2i::length() const {
	return Math::sqrt((double)length_squared());
}

Vector2i Vector2i::operator+(const Vector2i &p_v) const {
	return Vector2i(x + p_v.x, y + p_v.y);
}
void Vector2i::operator+=(const Vector2i &p_v) {
	x += p_v.x;
	y += p_v.y;
}
Vector2i Vector2i::operator-(const Vector2i &p_v) const {
	return Vector2i(x - p_v.x, y - p_v.y);
}
void Vector2i::operator-=(const Vector2i &p_v) {
	x -= p_v.x;
	y -= p_v.y;
}

Vector2i Vector2i::operator*(const Vector2i &p_v1) const {
	return Vector2i(x * p_v1.x, y * p_v1.y);
};

Vector2i Vector2i::operator*(const int &rvalue) const {
	return Vector2i(x * rvalue, y * rvalue);
};
void Vector2i::operator*=(const int &rvalue) {
	x *= rvalue;
	y *= rvalue;
};

Vector2i Vector2i::operator/(const Vector2i &p_v1) const {
	return Vector2i(x / p_v1.x, y / p_v1.y);
};

Vector2i Vector2i::operator/(const int &rvalue) const {
	return Vector2i(x / rvalue, y / rvalue);
};

void Vector2i::operator/=(const int &rvalue) {
	x /= rvalue;
	y /= rvalue;
};

Vector2i Vector2i::operator-() const {
	return Vector2i(-x, -y);
}

bool Vector2i::operator==(const Vector2i &p_vec2) const {
	return x == p_vec2.x && y == p_vec2.y;
}
bool Vector2i::operator!=(const Vector2i &p_vec2) const {
	return x != p_vec2.x || y != p_vec2.y;
}

Vector2i::operator String() const {
	return "(" + itos(x) + ", " + itos(y) + ")";
}
#line 0

#line 1 "sfw/core/rect2.cpp"
/*************************************************************************/
/*  rect2.cpp                                                            */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

bool Rect2::is_equal_approx(const Rect2 &p_rect) const {
	return position.is_equal_approx(p_rect.position) && size.is_equal_approx(p_rect.size);
}

bool Rect2::intersects_segment(const Point2 &p_from, const Point2 &p_to, Point2 *r_pos, Point2 *r_normal) const {
	real_t min = 0, max = 1;
	int axis = 0;
	real_t sign = 0;

	for (int i = 0; i < 2; i++) {
		real_t seg_from = p_from[i];
		real_t seg_to = p_to[i];
		real_t box_begin = position[i];
		real_t box_end = box_begin + size[i];
		real_t cmin, cmax;
		real_t csign;

		if (seg_from < seg_to) {
			if (seg_from > box_end || seg_to < box_begin) {
				return false;
			}
			real_t length = seg_to - seg_from;
			cmin = (seg_from < box_begin) ? ((box_begin - seg_from) / length) : 0;
			cmax = (seg_to > box_end) ? ((box_end - seg_from) / length) : 1;
			csign = -1.0;

		} else {
			if (seg_to > box_end || seg_from < box_begin) {
				return false;
			}
			real_t length = seg_to - seg_from;
			cmin = (seg_from > box_end) ? (box_end - seg_from) / length : 0;
			cmax = (seg_to < box_begin) ? (box_begin - seg_from) / length : 1;
			csign = 1.0;
		}

		if (cmin > min) {
			min = cmin;
			axis = i;
			sign = csign;
		}
		if (cmax < max) {
			max = cmax;
		}
		if (max < min) {
			return false;
		}
	}

	Vector2 rel = p_to - p_from;

	if (r_normal) {
		Vector2 normal;
		normal[axis] = sign;
		*r_normal = normal;
	}

	if (r_pos) {
		*r_pos = p_from + rel * min;
	}

	return true;
}

bool Rect2::intersects_transformed(const Transform2D &p_xform, const Rect2 &p_rect) const {
	//SAT intersection between local and transformed rect2

	Vector2 xf_points[4] = {
		p_xform.xform(p_rect.position),
		p_xform.xform(Vector2(p_rect.position.x + p_rect.size.x, p_rect.position.y)),
		p_xform.xform(Vector2(p_rect.position.x, p_rect.position.y + p_rect.size.y)),
		p_xform.xform(Vector2(p_rect.position.x + p_rect.size.x, p_rect.position.y + p_rect.size.y)),
	};

	real_t low_limit;

	//base rect2 first (faster)

	if (xf_points[0].y > position.y) {
		goto next1;
	}
	if (xf_points[1].y > position.y) {
		goto next1;
	}
	if (xf_points[2].y > position.y) {
		goto next1;
	}
	if (xf_points[3].y > position.y) {
		goto next1;
	}

	return false;

next1:

	low_limit = position.y + size.y;

	if (xf_points[0].y < low_limit) {
		goto next2;
	}
	if (xf_points[1].y < low_limit) {
		goto next2;
	}
	if (xf_points[2].y < low_limit) {
		goto next2;
	}
	if (xf_points[3].y < low_limit) {
		goto next2;
	}

	return false;

next2:

	if (xf_points[0].x > position.x) {
		goto next3;
	}
	if (xf_points[1].x > position.x) {
		goto next3;
	}
	if (xf_points[2].x > position.x) {
		goto next3;
	}
	if (xf_points[3].x > position.x) {
		goto next3;
	}

	return false;

next3:

	low_limit = position.x + size.x;

	if (xf_points[0].x < low_limit) {
		goto next4;
	}
	if (xf_points[1].x < low_limit) {
		goto next4;
	}
	if (xf_points[2].x < low_limit) {
		goto next4;
	}
	if (xf_points[3].x < low_limit) {
		goto next4;
	}

	return false;

next4:

	Vector2 xf_points2[4] = {
		position,
		Vector2(position.x + size.x, position.y),
		Vector2(position.x, position.y + size.y),
		Vector2(position.x + size.x, position.y + size.y),
	};

	real_t maxa = p_xform.columns[0].dot(xf_points2[0]);
	real_t mina = maxa;

	real_t dp = p_xform.columns[0].dot(xf_points2[1]);
	maxa = MAX(dp, maxa);
	mina = MIN(dp, mina);

	dp = p_xform.columns[0].dot(xf_points2[2]);
	maxa = MAX(dp, maxa);
	mina = MIN(dp, mina);

	dp = p_xform.columns[0].dot(xf_points2[3]);
	maxa = MAX(dp, maxa);
	mina = MIN(dp, mina);

	real_t maxb = p_xform.columns[0].dot(xf_points[0]);
	real_t minb = maxb;

	dp = p_xform.columns[0].dot(xf_points[1]);
	maxb = MAX(dp, maxb);
	minb = MIN(dp, minb);

	dp = p_xform.columns[0].dot(xf_points[2]);
	maxb = MAX(dp, maxb);
	minb = MIN(dp, minb);

	dp = p_xform.columns[0].dot(xf_points[3]);
	maxb = MAX(dp, maxb);
	minb = MIN(dp, minb);

	if (mina > maxb) {
		return false;
	}
	if (minb > maxa) {
		return false;
	}

	maxa = p_xform.columns[1].dot(xf_points2[0]);
	mina = maxa;

	dp = p_xform.columns[1].dot(xf_points2[1]);
	maxa = MAX(dp, maxa);
	mina = MIN(dp, mina);

	dp = p_xform.columns[1].dot(xf_points2[2]);
	maxa = MAX(dp, maxa);
	mina = MIN(dp, mina);

	dp = p_xform.columns[1].dot(xf_points2[3]);
	maxa = MAX(dp, maxa);
	mina = MIN(dp, mina);

	maxb = p_xform.columns[1].dot(xf_points[0]);
	minb = maxb;

	dp = p_xform.columns[1].dot(xf_points[1]);
	maxb = MAX(dp, maxb);
	minb = MIN(dp, minb);

	dp = p_xform.columns[1].dot(xf_points[2]);
	maxb = MAX(dp, maxb);
	minb = MIN(dp, minb);

	dp = p_xform.columns[1].dot(xf_points[3]);
	maxb = MAX(dp, maxb);
	minb = MIN(dp, minb);

	if (mina > maxb) {
		return false;
	}
	if (minb > maxa) {
		return false;
	}

	return true;
}

Rect2::operator String() const {
	return "[P: " + position.operator String() + ", S: " + size + "]";
}
#line 0

#line 1 "sfw/core/rect2i.cpp"
/*************************************************************************/
/*  rect2i.cpp                                                           */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

Rect2i::operator String() const {
	return "[P: " + position.operator String() + ", S: " + size + "]";
}
#line 0

#line 1 "sfw/core/vector4.cpp"
/*************************************************************************/
/*  vector4.cpp                                                          */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

void Vector4::set_axis(const int p_axis, const real_t p_value) {
	ERR_FAIL_INDEX(p_axis, 4);
	components[p_axis] = p_value;
}

real_t Vector4::get_axis(const int p_axis) const {
	ERR_FAIL_INDEX_V(p_axis, 4, 0);
	return operator[](p_axis);
}

Vector4::Axis Vector4::min_axis() const {
	uint32_t min_index = 0;
	real_t min_value = x;
	for (uint32_t i = 1; i < 4; i++) {
		if (operator[](i) <= min_value) {
			min_index = i;
			min_value = operator[](i);
		}
	}
	return Vector4::Axis(min_index);
}

Vector4::Axis Vector4::max_axis() const {
	uint32_t max_index = 0;
	real_t max_value = x;
	for (uint32_t i = 1; i < 4; i++) {
		if (operator[](i) > max_value) {
			max_index = i;
			max_value = operator[](i);
		}
	}
	return Vector4::Axis(max_index);
}

bool Vector4::is_equal_approx(const Vector4 &p_vec4) const {
	return Math::is_equal_approx(x, p_vec4.x) && Math::is_equal_approx(y, p_vec4.y) && Math::is_equal_approx(z, p_vec4.z) && Math::is_equal_approx(w, p_vec4.w);
}

real_t Vector4::length() const {
	return Math::sqrt(length_squared());
}

void Vector4::normalize() {
	*this /= length();
}

Vector4 Vector4::normalized() const {
	return *this / length();
}

bool Vector4::is_normalized() const {
	return Math::is_equal_approx(length_squared(), 1, (real_t)UNIT_EPSILON); // Use less epsilon.
}

Vector4 Vector4::limit_length(const real_t p_len) const {
	const real_t l = length();
	Vector4 v = *this;
	if (l > 0 && p_len < l) {
		v /= l;
		v *= p_len;
	}

	return v;
}

real_t Vector4::distance_to(const Vector4 &p_to) const {
	return (p_to - *this).length();
}

Vector4 Vector4::direction_to(const Vector4 &p_to) const {
	Vector4 ret(p_to.x - x, p_to.y - y, p_to.z - z, p_to.w - w);
	ret.normalize();
	return ret;
}

real_t Vector4::distance_squared_to(const Vector4 &p_to) const {
	return (p_to - *this).length_squared();
}

Vector4 Vector4::abs() const {
	return Vector4(Math::abs(x), Math::abs(y), Math::abs(z), Math::abs(w));
}

Vector4 Vector4::sign() const {
	return Vector4(SGN(x), SGN(y), SGN(z), SGN(w));
}

Vector4 Vector4::floor() const {
	return Vector4(Math::floor(x), Math::floor(y), Math::floor(z), Math::floor(w));
}

Vector4 Vector4::ceil() const {
	return Vector4(Math::ceil(x), Math::ceil(y), Math::ceil(z), Math::ceil(w));
}

Vector4 Vector4::round() const {
	return Vector4(Math::round(x), Math::round(y), Math::round(z), Math::round(w));
}

Vector4 Vector4::linear_interpolate(const Vector4 &p_to, const real_t p_weight) const {
	return Vector4(
			x + (p_weight * (p_to.x - x)),
			y + (p_weight * (p_to.y - y)),
			z + (p_weight * (p_to.z - z)),
			w + (p_weight * (p_to.w - w)));
}

Vector4 Vector4::cubic_interpolate(const Vector4 &p_b, const Vector4 &p_pre_a, const Vector4 &p_post_b, const real_t p_weight) const {
	Vector4 res = *this;
	res.x = Math::cubic_interpolate(res.x, p_b.x, p_pre_a.x, p_post_b.x, p_weight);
	res.y = Math::cubic_interpolate(res.y, p_b.y, p_pre_a.y, p_post_b.y, p_weight);
	res.z = Math::cubic_interpolate(res.z, p_b.z, p_pre_a.z, p_post_b.z, p_weight);
	res.w = Math::cubic_interpolate(res.w, p_b.w, p_pre_a.w, p_post_b.w, p_weight);
	return res;
}

Vector4 Vector4::posmod(const real_t p_mod) const {
	return Vector4(Math::fposmod(x, p_mod), Math::fposmod(y, p_mod), Math::fposmod(z, p_mod), Math::fposmod(w, p_mod));
}

Vector4 Vector4::posmodv(const Vector4 &p_modv) const {
	return Vector4(Math::fposmod(x, p_modv.x), Math::fposmod(y, p_modv.y), Math::fposmod(z, p_modv.z), Math::fposmod(w, p_modv.w));
}

void Vector4::snap(const Vector4 &p_step) {
	x = Math::stepify(x, p_step.x);
	y = Math::stepify(y, p_step.y);
	z = Math::stepify(z, p_step.z);
	w = Math::stepify(w, p_step.w);
}

Vector4 Vector4::snapped(const Vector4 &p_step) const {
	Vector4 v = *this;
	v.snap(p_step);
	return v;
}

Vector4 Vector4::inverse() const {
	return Vector4(1.0f / x, 1.0f / y, 1.0f / z, 1.0f / w);
}

Vector4 Vector4::clamp(const Vector4 &p_min, const Vector4 &p_max) const {
	return Vector4(
			CLAMP(x, p_min.x, p_max.x),
			CLAMP(y, p_min.y, p_max.y),
			CLAMP(z, p_min.z, p_max.z),
			CLAMP(w, p_min.w, p_max.w));
}

Vector4::operator String() const {
	return "(" + String::num_real(x) + ", " + String::num_real(y) + ", " + String::num_real(z) + ", " + String::num_real(w) + ")";
}
#line 0

#line 1 "sfw/core/file_access.cpp"

/*************************************************************************/
/*  file_access.cpp                                                      */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

#if defined(_WIN64) || defined(_WIN32)

#include <share.h> // _SH_DENYNO
#include <shlwapi.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <tchar.h>
#include <wchar.h>

#ifdef _MSC_VER
#define S_ISREG(m) ((m)&_S_IFREG)
#endif

#else

#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <errno.h>

#include <unistd.h>

#ifdef MSVC
#define S_ISREG(m) ((m)&_S_IFREG)
#include <io.h>
#endif
#ifndef S_ISREG
#define S_ISREG(m) ((m)&S_IFREG)
#endif

#ifndef NO_FCNTL
#include <fcntl.h>
#else
#include <sys/ioctl.h>
#endif

#endif

#if defined(_WIN64) || defined(_WIN32)

void FileAccess::check_errors() const {
	ERR_FAIL_COND(!f);

	if (feof(f)) {
		last_error = ERR_FILE_EOF;
	}
}

Error FileAccess::_open(const String &p_path, int p_mode_flags) {
	path_src = p_path;
	path = fix_path(p_path);

	if (f) {
		close();
	}

	const WCHAR *mode_string;

	if (p_mode_flags == READ) {
		mode_string = L"rb";
	} else if (p_mode_flags == WRITE) {
		mode_string = L"wb";
	} else if (p_mode_flags == READ_WRITE) {
		mode_string = L"rb+";
	} else if (p_mode_flags == WRITE_READ) {
		mode_string = L"wb+";
	} else {
		return ERR_INVALID_PARAMETER;
	}

	/* pretty much every implementation that uses fopen as primary
	   backend supports utf8 encoding */

	struct _stat st;
	if (_wstat((LPCWSTR)(path.utf16().get_data()), &st) == 0) {
		if (!S_ISREG(st.st_mode)) {
			return ERR_FILE_CANT_OPEN;
		}
	};

#ifdef TOOLS_ENABLED
	// Windows is case insensitive, but all other platforms are sensitive to it
	// To ease cross-platform development, we issue a warning if users try to access
	// a file using the wrong case (which *works* on Windows, but won't on other
	// platforms).
	if (p_mode_flags == READ) {
		WIN32_FIND_DATAW d;
		HANDLE f = FindFirstFileW((LPCWSTR)(path.utf16().get_data()), &d);

		if (f != INVALID_HANDLE_VALUE) {
			String fname = String::utf16((const char16_t *)(d.cFileName));

			if (fname != String()) {
				String base_file = path.get_file();
				if (base_file != fname && base_file.findn(fname) == 0) {
					LOG_WARN("Case mismatch opening requested file '" + base_file + "', stored as '" + fname + "' in the filesystem. This file will not open when exported to other case-sensitive platforms.");
				}
			}

			FindClose(f);
		}
	}
#endif

	if (is_backup_save_enabled() && p_mode_flags & WRITE && !(p_mode_flags & READ)) {
		save_path = path;
		path = path + ".tmp";
	}

	f = _wfsopen((LPCWSTR)(path.utf16().get_data()), mode_string, _SH_DENYNO);

	if (f == nullptr) {
		switch (errno) {
			case ENOENT: {
				last_error = ERR_FILE_NOT_FOUND;
			} break;
			default: {
				last_error = ERR_FILE_CANT_OPEN;
			} break;
		}
		return last_error;
	} else {
		last_error = OK;
		flags = p_mode_flags;
		return OK;
	}
}

void FileAccess::close() {
	if (!f) {
		return;
	}

	fclose(f);
	f = NULL;

	if (save_path != "") {
		bool rename_error = true;
		int attempts = 4;
		while (rename_error && attempts) {
			// This workaround of trying multiple times is added to deal with paranoid Windows
			// antiviruses that love reading just written files even if they are not executable, thus
			// locking the file and preventing renaming from happening.

#ifdef UWP_ENABLED
			// UWP has no PathFileExists, so we check attributes instead
			DWORD fileAttr;

			fileAttr = GetFileAttributesW((LPCWSTR)(save_path.utf16().get_data()));
			if (INVALID_FILE_ATTRIBUTES == fileAttr) {
#else
			if (!PathFileExistsW((LPCWSTR)(save_path.utf16().get_data()))) {
#endif
				//creating new file
				rename_error = _wrename((LPCWSTR)((save_path + ".tmp").utf16().get_data()), (LPCWSTR)(save_path.utf16().get_data())) != 0;
			} else {
				//atomic replace for existing file
				rename_error = !ReplaceFileW((LPCWSTR)(save_path.utf16().get_data()), (LPCWSTR)((save_path + ".tmp").utf16().get_data()), NULL, 2 | 4, NULL, NULL);
			}
			if (rename_error) {
				attempts--;
				SFWTime::sleep_us(100000); // wait 100msec and try again
			}
		}

		if (rename_error) {
			if (close_fail_notify) {
				close_fail_notify(save_path);
			}
		}

		save_path = "";

		ERR_FAIL_COND_MSG(rename_error, "Safe save failed. This may be a permissions problem, but also may happen because you are running a paranoid antivirus. If this is the case, please switch to Windows Defender or disable the 'safe save' option in editor settings. This makes it work, but increases the risk of file corruption in a crash.");
	}
}

String FileAccess::get_path() const {
	return path_src;
}

String FileAccess::get_path_absolute() const {
	return path;
}

bool FileAccess::is_open() const {
	return (f != NULL);
}

void FileAccess::seek(uint64_t p_position) {
	ERR_FAIL_COND(!f);

	last_error = OK;

	if (_fseeki64(f, p_position, SEEK_SET)) {
		check_errors();
	}

	prev_op = 0;
}

void FileAccess::seek_end(int64_t p_position) {
	ERR_FAIL_COND(!f);

	if (_fseeki64(f, p_position, SEEK_END)) {
		check_errors();
	}

	prev_op = 0;
}

uint64_t FileAccess::get_position() const {
	int64_t aux_position = _ftelli64(f);

	if (aux_position < 0) {
		check_errors();
	}

	return aux_position;
}

uint64_t FileAccess::get_len() const {
	ERR_FAIL_COND_V(!f, 0);

	uint64_t pos = get_position();
	_fseeki64(f, 0, SEEK_END);
	uint64_t size = get_position();
	_fseeki64(f, pos, SEEK_SET);

	return size;
}

bool FileAccess::eof_reached() const {
	check_errors();
	return last_error == ERR_FILE_EOF;
}

uint8_t FileAccess::get_8() const {
	ERR_FAIL_COND_V(!f, 0);
	if (flags == READ_WRITE || flags == WRITE_READ) {
		if (prev_op == WRITE) {
			fflush(f);
		}
		prev_op = READ;
	}
	uint8_t b;
	if (fread(&b, 1, 1, f) == 0) {
		check_errors();
		b = '\0';
	};

	return b;
}

uint64_t FileAccess::get_buffer(uint8_t *p_dst, uint64_t p_length) const {
	ERR_FAIL_COND_V(!p_dst && p_length > 0, -1);
	ERR_FAIL_COND_V(!f, -1);

	if (flags == READ_WRITE || flags == WRITE_READ) {
		if (prev_op == WRITE) {
			fflush(f);
		}
		prev_op = READ;
	}
	uint64_t read = fread(p_dst, 1, p_length, f);
	check_errors();
	return read;
};

Error FileAccess::get_error() const {
	return last_error;
}

void FileAccess::flush() {
	ERR_FAIL_COND(!f);
	fflush(f);

	if (prev_op == WRITE) {
		prev_op = 0;
	}
}

void FileAccess::store_8(uint8_t p_dest) {
	ERR_FAIL_COND(!f);
	if (flags == READ_WRITE || flags == WRITE_READ) {
		if (prev_op == READ) {
			if (last_error != ERR_FILE_EOF) {
				fseek(f, 0, SEEK_CUR);
			}
		}
		prev_op = WRITE;
	}
	fwrite(&p_dest, 1, 1, f);
}

void FileAccess::store_buffer(const uint8_t *p_src, uint64_t p_length) {
	ERR_FAIL_COND(!f);
	ERR_FAIL_COND(!p_src && p_length > 0);

	if (flags == READ_WRITE || flags == WRITE_READ) {
		if (prev_op == READ) {
			if (last_error != ERR_FILE_EOF) {
				fseek(f, 0, SEEK_CUR);
			}
		}
		prev_op = WRITE;
	}
	ERR_FAIL_COND(fwrite(p_src, 1, p_length, f) != (size_t)p_length);
}

bool FileAccess::file_exists(const String &p_name) {
	String filename = fix_path(p_name);

	FILE *g = _wfsopen((LPCWSTR)(filename.utf16().get_data()), L"rb", _SH_DENYNO);

	if (g == nullptr) {
		return false;
	} else {
		fclose(g);
		return true;
	}
}

uint64_t FileAccess::_get_modified_time(const String &p_file) {
	String file = fix_path(p_file);
	if (file.ends_with("/") && file != "/")
		file = file.substr(0, file.length() - 1);

	struct _stat st;
	int rv = _wstat((LPCWSTR)(file.utf16().get_data()), &st);

	if (rv == 0) {
		return st.st_mtime;
	} else {
		LOG_TRACE("Failed to get modified time for: " + p_file + "");
		return 0;
	}
}

uint32_t FileAccess::_get_unix_permissions(const String &p_file) {
	return 0;
}

Error FileAccess::_set_unix_permissions(const String &p_file, uint32_t p_permissions) {
	return ERR_UNAVAILABLE;
}

FileAccess::FileAccess() :
		f(NULL),
		flags(0),
		prev_op(0),
		last_error(OK) {
}
FileAccess::~FileAccess() {
	close();
}

#else

void FileAccess::check_errors() const {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use.");

	if (feof(f)) {
		last_error = ERR_FILE_EOF;
	}
}

Error FileAccess::_open(const String &p_path, int p_mode_flags) {
	if (f) {
		fclose(f);
	}
	f = nullptr;

	path_src = p_path;
	path = fix_path(p_path);
	//printf("opening %s, %i\n", path.utf8().get_data(), Memory::get_static_mem_usage());

	ERR_FAIL_COND_V_MSG(f, ERR_ALREADY_IN_USE, "File is already in use.");
	const char *mode_string;

	if (p_mode_flags == READ) {
		mode_string = "rb";
	} else if (p_mode_flags == WRITE) {
		mode_string = "wb";
	} else if (p_mode_flags == READ_WRITE) {
		mode_string = "rb+";
	} else if (p_mode_flags == WRITE_READ) {
		mode_string = "wb+";
	} else {
		return ERR_INVALID_PARAMETER;
	}

	/* pretty much every implementation that uses fopen as primary
	   backend (unix-compatible mostly) supports utf8 encoding */

	//printf("opening %s as %s\n", p_path.utf8().get_data(), path.utf8().get_data());
	struct stat st;
	int err = stat(path.utf8().get_data(), &st);
	if (!err) {
		switch (st.st_mode & S_IFMT) {
			case S_IFLNK:
			case S_IFREG:
				break;
			default:
				return ERR_FILE_CANT_OPEN;
		}
	}

	if (is_backup_save_enabled() && (p_mode_flags & WRITE) && !(p_mode_flags & READ)) {
		save_path = path;
		path = path + ".tmp";
	}

	f = fopen(path.utf8().get_data(), mode_string);

	if (f == nullptr) {
		switch (errno) {
			case ENOENT: {
				last_error = ERR_FILE_NOT_FOUND;
			} break;
			default: {
				last_error = ERR_FILE_CANT_OPEN;
			} break;
		}
		return last_error;
	}

	// Set close on exec to avoid leaking it to subprocesses.
	int fd = fileno(f);

	if (fd != -1) {
#if defined(NO_FCNTL)
		unsigned long par = 0;
		ioctl(fd, FIOCLEX, &par);
#else
		int opts = fcntl(fd, F_GETFD);
		fcntl(fd, F_SETFD, opts | FD_CLOEXEC);
#endif
	}

	last_error = OK;
	flags = p_mode_flags;
	return OK;
}

void FileAccess::close() {
	if (!f) {
		return;
	}

	fclose(f);
	f = nullptr;

	if (close_notification_func) {
		close_notification_func(path, flags);
	}

	if (save_path != "") {
		int rename_error = rename((save_path + ".tmp").utf8().get_data(), save_path.utf8().get_data());

		if (rename_error && close_fail_notify) {
			close_fail_notify(save_path);
		}

		save_path = "";
		ERR_FAIL_COND(rename_error != 0);
	}
}

bool FileAccess::is_open() const {
	return (f != nullptr);
}

String FileAccess::get_path() const {
	return path_src;
}

String FileAccess::get_path_absolute() const {
	return path;
}

void FileAccess::seek(uint64_t p_position) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use.");

	last_error = OK;
	if (fseeko(f, p_position, SEEK_SET)) {
		check_errors();
	}
}

void FileAccess::seek_end(int64_t p_position) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use.");

	if (fseeko(f, p_position, SEEK_END)) {
		check_errors();
	}
}

uint64_t FileAccess::get_position() const {
	ERR_FAIL_COND_V_MSG(!f, 0, "File must be opened before use.");

	int64_t pos = ftello(f);
	if (pos < 0) {
		check_errors();
		ERR_FAIL_V(0);
	}
	return pos;
}

uint64_t FileAccess::get_len() const {
	ERR_FAIL_COND_V_MSG(!f, 0, "File must be opened before use.");

	int64_t pos = ftello(f);
	ERR_FAIL_COND_V(pos < 0, 0);
	ERR_FAIL_COND_V(fseeko(f, 0, SEEK_END), 0);
	int64_t size = ftello(f);
	ERR_FAIL_COND_V(size < 0, 0);
	ERR_FAIL_COND_V(fseeko(f, pos, SEEK_SET), 0);

	return size;
}

bool FileAccess::eof_reached() const {
	return last_error == ERR_FILE_EOF;
}

uint8_t FileAccess::get_8() const {
	ERR_FAIL_COND_V_MSG(!f, 0, "File must be opened before use.");
	uint8_t b;
	if (fread(&b, 1, 1, f) == 0) {
		check_errors();
		b = '\0';
	}
	return b;
}

uint64_t FileAccess::get_buffer(uint8_t *p_dst, uint64_t p_length) const {
	ERR_FAIL_COND_V(!p_dst && p_length > 0, -1);
	ERR_FAIL_COND_V_MSG(!f, -1, "File must be opened before use.");

	uint64_t read = fread(p_dst, 1, p_length, f);
	check_errors();
	return read;
};

Error FileAccess::get_error() const {
	return last_error;
}

void FileAccess::flush() {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use.");
	fflush(f);
}

void FileAccess::store_8(uint8_t p_dest) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use.");
	ERR_FAIL_COND(fwrite(&p_dest, 1, 1, f) != 1);
}

void FileAccess::store_buffer(const uint8_t *p_src, uint64_t p_length) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use.");
	ERR_FAIL_COND(!p_src && p_length > 0);
	ERR_FAIL_COND(fwrite(p_src, 1, p_length, f) != p_length);
}

bool FileAccess::file_exists(const String &p_path) {
	int err;
	struct stat st;
	String filename = fix_path(p_path);

	// Does the name exist at all?
	err = stat(filename.utf8().get_data(), &st);
	if (err) {
		return false;
	}

#if defined(_WIN64) || defined(_WIN32)
	if (_access(filename.utf8().get_data(), 4) == -1) {
		return false;
	}
#else
	// See if we have access to the file
	if (access(filename.utf8().get_data(), F_OK)) {
		return false;
	}
#endif

	// See if this is a regular file
	switch (st.st_mode & S_IFMT) {
		case S_IFLNK:
		case S_IFREG:
			return true;
		default:
			return false;
	}
}

uint64_t FileAccess::_get_modified_time(const String &p_file) {
	String file = fix_path(p_file);
	struct stat flags;
	int err = stat(file.utf8().get_data(), &flags);

	if (!err) {
		return flags.st_mtime;
	} else {
		LOG_TRACE("Failed to get modified time for: " + p_file + "");
		return 0;
	};
}

uint32_t FileAccess::_get_unix_permissions(const String &p_file) {
	String file = fix_path(p_file);
	struct stat flags;
	int err = stat(file.utf8().get_data(), &flags);

	if (!err) {
		return flags.st_mode & 0x7FF; //only permissions
	} else {
		ERR_FAIL_V_MSG(0, "Failed to get unix permissions for: " + p_file + ".");
	};
}

Error FileAccess::_set_unix_permissions(const String &p_file, uint32_t p_permissions) {
	String file = fix_path(p_file);

	int err = chmod(file.utf8().get_data(), p_permissions);
	if (!err) {
		return OK;
	}

	return FAILED;
}

FileCloseNotificationFunc FileAccess::close_notification_func = nullptr;

FileAccess::FileAccess() :
		f(nullptr),
		flags(0),
		last_error(OK) {
	endian_swap = false;
	real_is_double = false;
}

FileAccess::~FileAccess() {
	close();
}

#endif

FileAccess::FileCloseFailNotify FileAccess::close_fail_notify = nullptr;

bool FileAccess::backup_save = false;

FileAccess *FileAccess::create() {
	return memnew(FileAccess());
}

bool FileAccess::exists(const String &p_name) {
	FileAccess *f = create_and_open(p_name, READ);
	if (!f) {
		return false;
	}
	memdelete(f);
	return true;
}

Error FileAccess::reopen(const String &p_path, int p_mode_flags) {
	return _open(p_path, p_mode_flags);
};

Error FileAccess::open(const String &p_path, int p_mode_flags) {
	return _open(p_path, p_mode_flags);
}

FileAccess *FileAccess::create_and_open(const String &p_path, int p_mode_flags, Error *r_error) {
	//try packed data first

	FileAccess *ret = nullptr;

	ret = create();
	Error err = ret->_open(p_path, p_mode_flags);

	if (r_error) {
		*r_error = err;
	}
	if (err != OK) {
		memdelete(ret);
		ret = nullptr;
	}

	return ret;
}

String FileAccess::fix_path(const String &p_path) const {
	//helper used by file accesses that use a single filesystem

	String r_path = p_path.replace("\\", "/");

	return r_path;
}

/* these are all implemented for ease of porting, then can later be optimized */

uint16_t FileAccess::get_16() const {
	uint16_t res;
	uint8_t a, b;

	a = get_8();
	b = get_8();

	if (endian_swap) {
		SWAP(a, b);
	}

	res = b;
	res <<= 8;
	res |= a;

	return res;
}
uint32_t FileAccess::get_32() const {
	uint32_t res;
	uint16_t a, b;

	a = get_16();
	b = get_16();

	if (endian_swap) {
		SWAP(a, b);
	}

	res = b;
	res <<= 16;
	res |= a;

	return res;
}
uint64_t FileAccess::get_64() const {
	uint64_t res;
	uint32_t a, b;

	a = get_32();
	b = get_32();

	if (endian_swap) {
		SWAP(a, b);
	}

	res = b;
	res <<= 32;
	res |= a;

	return res;
}

float FileAccess::get_float() const {
	MarshallFloat m;
	m.i = get_32();
	return m.f;
};

real_t FileAccess::get_real() const {
	if (real_is_double) {
		return get_double();
	} else {
		return get_float();
	}
}

double FileAccess::get_double() const {
	MarshallDouble m;
	m.l = get_64();
	return m.d;
};

String FileAccess::get_token() const {
	CharString token;

	CharType c = get_8();

	while (!eof_reached()) {
		if (c <= ' ') {
			if (token.length()) {
				break;
			}
		} else {
			token += c;
		}
		c = get_8();
	}

	return String::utf8(token.get_data());
}

class CharBuffer {
	Vector<char> vector;
	char stack_buffer[256];

	char *buffer;
	int capacity;
	int written;

	bool grow() {
		if (vector.resize(next_power_of_2(1 + written)) != OK) {
			return false;
		}

		if (buffer == stack_buffer) { // first chunk?

			for (int i = 0; i < written; i++) {
				vector.write[i] = stack_buffer[i];
			}
		}

		buffer = vector.ptrw();
		capacity = vector.size();
		ERR_FAIL_COND_V(written >= capacity, false);

		return true;
	}

public:
	_FORCE_INLINE_ CharBuffer() :
			buffer(stack_buffer),
			capacity(sizeof(stack_buffer) / sizeof(char)),
			written(0) {
	}

	_FORCE_INLINE_ void push_back(char c) {
		if (written >= capacity) {
			ERR_FAIL_COND(!grow());
		}

		buffer[written++] = c;
	}

	_FORCE_INLINE_ const char *get_data() const {
		return buffer;
	}
};

String FileAccess::get_line() const {
	CharBuffer line;

	CharType c = get_8();

	while (!eof_reached()) {
		if (c == '\n' || c == '\0') {
			line.push_back(0);
			return String::utf8(line.get_data());
		} else if (c != '\r') {
			line.push_back(c);
		}

		c = get_8();
	}
	line.push_back(0);
	return String::utf8(line.get_data());
}

Vector<String> FileAccess::get_csv_line(const String &p_delim) const {
	ERR_FAIL_COND_V_MSG(p_delim.length() != 1, Vector<String>(), "Only single character delimiters are supported to parse CSV lines.");
	ERR_FAIL_COND_V_MSG(p_delim[0] == '"', Vector<String>(), "The double quotation mark character (\") is not supported as a delimiter for CSV lines.");

	String line;

	// CSV can support entries with line breaks as long as they are enclosed
	// in double quotes. So our "line" might be more than a single line in the
	// text file.
	int qc = 0;
	do {
		if (eof_reached()) {
			break;
		}
		line += get_line() + "\n";
		qc = 0;
		for (int i = 0; i < line.length(); i++) {
			if (line[i] == '"') {
				qc++;
			}
		}
	} while (qc % 2);

	// Remove the extraneous newline we've added above.
	line = line.substr(0, line.length() - 1);

	Vector<String> strings;

	bool in_quote = false;
	String current;
	for (int i = 0; i < line.length(); i++) {
		CharType c = line[i];
		// A delimiter ends the current entry, unless it's in a quoted string.
		if (!in_quote && c == p_delim[0]) {
			strings.push_back(current);
			current = String();
		} else if (c == '"') {
			// Doubled quotes are escapes for intentional quotes in the string.
			if (line[i + 1] == '"' && in_quote) {
				current += '"';
				i++;
			} else {
				in_quote = !in_quote;
			}
		} else {
			current += c;
		}
	}
	strings.push_back(current);

	return strings;
}

/*
uint64_t FileAccess::get_buffer(uint8_t *p_dst, uint64_t p_length) const {
	ERR_FAIL_COND_V(!p_dst && p_length > 0, -1);

	uint64_t i = 0;
	for (i = 0; i < p_length && !eof_reached(); i++) {
		p_dst[i] = get_8();
	}

	return i;
}
*/

String FileAccess::get_as_utf8_string(bool p_skip_cr) const {
	Vector<uint8_t> sourcef;
	uint64_t len = get_len();
	sourcef.resize(len + 1);

	uint8_t *w = sourcef.ptrw();
	uint64_t r = get_buffer(w, len);
	ERR_FAIL_COND_V(r != len, String());
	w[len] = 0;

	String s;
	if (s.parse_utf8((const char *)w, -1, p_skip_cr)) {
		return String();
	}

	return s;
}

void FileAccess::store_16(uint16_t p_dest) {
	uint8_t a, b;

	a = p_dest & 0xFF;
	b = p_dest >> 8;

	if (endian_swap) {
		SWAP(a, b);
	}

	store_8(a);
	store_8(b);
}
void FileAccess::store_32(uint32_t p_dest) {
	uint16_t a, b;

	a = p_dest & 0xFFFF;
	b = p_dest >> 16;

	if (endian_swap) {
		SWAP(a, b);
	}

	store_16(a);
	store_16(b);
}
void FileAccess::store_64(uint64_t p_dest) {
	uint32_t a, b;

	a = p_dest & 0xFFFFFFFF;
	b = p_dest >> 32;

	if (endian_swap) {
		SWAP(a, b);
	}

	store_32(a);
	store_32(b);
}

void FileAccess::store_real(real_t p_real) {
	if (sizeof(real_t) == 4) {
		store_float(p_real);
	} else {
		store_double(p_real);
	}
}

void FileAccess::store_float(float p_dest) {
	MarshallFloat m;
	m.f = p_dest;
	store_32(m.i);
};

void FileAccess::store_double(double p_dest) {
	MarshallDouble m;
	m.d = p_dest;
	store_64(m.l);
};

uint64_t FileAccess::get_modified_time(const String &p_file) {
	FileAccess *fa = create();
	ERR_FAIL_COND_V_MSG(!fa, 0, "Cannot create FileAccess for path '" + p_file + "'.");

	uint64_t mt = fa->_get_modified_time(p_file);
	memdelete(fa);
	return mt;
}

uint32_t FileAccess::get_unix_permissions(const String &p_file) {
	FileAccess *fa = create();
	ERR_FAIL_COND_V_MSG(!fa, 0, "Cannot create FileAccess for path '" + p_file + "'.");

	uint32_t mt = fa->_get_unix_permissions(p_file);
	memdelete(fa);
	return mt;
}

Error FileAccess::set_unix_permissions(const String &p_file, uint32_t p_permissions) {
	FileAccess *fa = create();
	ERR_FAIL_COND_V_MSG(!fa, ERR_CANT_CREATE, "Cannot create FileAccess for path '" + p_file + "'.");

	Error err = fa->_set_unix_permissions(p_file, p_permissions);
	memdelete(fa);
	return err;
}

void FileAccess::store_string(const String &p_string) {
	if (p_string.length() == 0) {
		return;
	}

	CharString cs = p_string.utf8();
	store_buffer((uint8_t *)&cs[0], cs.length());
}

void FileAccess::store_pascal_string(const String &p_string) {
	CharString cs = p_string.utf8();
	store_32(cs.length());
	store_buffer((uint8_t *)&cs[0], cs.length());
};

String FileAccess::get_pascal_string() {
	uint32_t sl = get_32();
	CharString cs;
	cs.resize(sl + 1);
	get_buffer((uint8_t *)cs.ptr(), sl);
	cs[sl] = 0;

	String ret;
	ret.parse_utf8(cs.ptr());

	return ret;
};

void FileAccess::store_line(const String &p_line) {
	store_string(p_line);
	store_8('\n');
}

void FileAccess::store_csv_line(const Vector<String> &p_values, const String &p_delim) {
	ERR_FAIL_COND(p_delim.length() != 1);

	String line = "";
	int size = p_values.size();
	for (int i = 0; i < size; ++i) {
		String value = p_values[i];

		if (value.find("\"") != -1 || value.find(p_delim) != -1 || value.find("\n") != -1) {
			value = "\"" + value.replace("\"", "\"\"") + "\"";
		}
		if (i < size - 1) {
			value += p_delim;
		}

		line += value;
	}

	store_line(line);
}

void FileAccess::store_buffer_vec(const Vector<uint8_t> &data) {
	store_buffer(data.ptr(), data.size());
}

/*
void FileAccess::store_buffer(const uint8_t *p_src, uint64_t p_length) {
	ERR_FAIL_COND(!p_src && p_length > 0);
	for (uint64_t i = 0; i < p_length; i++) {
		store_8(p_src[i]);
	}
}
*/

Vector<uint8_t> FileAccess::get_file_as_array(const String &p_path, Error *r_error) {
	FileAccess *f = FileAccess::create_and_open(p_path, READ, r_error);
	if (!f) {
		if (r_error) { // if error requested, do not throw error
			return Vector<uint8_t>();
		}
		ERR_FAIL_V_MSG(Vector<uint8_t>(), "Can't open file from path '" + String(p_path) + "'.");
	}
	Vector<uint8_t> data;
	data.resize(f->get_len());
	f->get_buffer(data.ptrw(), data.size());
	memdelete(f);
	return data;
}

String FileAccess::get_file_as_string(const String &p_path, Error *r_error) {
	Error err;
	Vector<uint8_t> array = get_file_as_array(p_path, &err);
	if (r_error) {
		*r_error = err;
	}
	if (err != OK) {
		if (r_error) {
			return String();
		}
		ERR_FAIL_V_MSG(String(), "Can't get file as string from path '" + String(p_path) + "'.");
	}

	String ret;
	ret.parse_utf8((const char *)array.ptr(), array.size());
	return ret;
}

void FileAccess::write_file(const String &p_path, const String &data, Error *r_error) {
	FileAccess *f = FileAccess::create_and_open(p_path, WRITE, r_error);
	if (!f) {
		if (r_error) { // if error requested, do not throw error
			return;
		}

		ERR_FAIL_MSG("Can't open file from path '" + String(p_path) + "'.");
	}

	f->store_string(data);
	f->close();
	memdelete(f);
}
void FileAccess::write_file_buffer(const String &p_path, const Vector<uint8_t> &data, Error *r_error) {
	FileAccess *f = FileAccess::create_and_open(p_path, WRITE, r_error);
	if (!f) {
		if (r_error) { // if error requested, do not throw error
			return;
		}

		ERR_FAIL_MSG("Can't open file from path '" + String(p_path) + "'.");
	}

	f->store_buffer_vec(data);
	f->close();
	memdelete(f);
}

/*
FileAccess::FileAccess() {
	endian_swap = false;
	real_is_double = false;
};
*/
#line 0

#line 1 "sfw/core/dir_access.cpp"

/*************************************************************************/
/*  dir_access.cpp                                                       */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

#if defined(_WIN64) || defined(_WIN32)

#include <stdio.h>
#include <wchar.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#else

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef HAVE_MNTENT
#include <mntent.h>
#endif

#endif

#ifdef _MSC_VER
#pragma comment(lib, "Shlwapi")
#endif

#if defined(_WIN64) || defined(_WIN32)

bool DirAccess::is_link(String p_file) {
	return false;
};
String DirAccess::read_link(String p_file) {
	return p_file;
};
Error DirAccess::create_link(String p_source, String p_target) {
	return FAILED;
};

/*

[03:57] <reduz> yessopie, so i don't havemak to rely on unicows
[03:58] <yessopie> reduz- yeah, all of the functions fail, and then you can call GetLastError () which will return 120
[03:58] <drumstick> CategoryApl, hehe, what? :)
[03:59] <CategoryApl> didn't Verona lead to some trouble
[03:59] <yessopie> 120 = ERROR_CALL_NOT_IMPLEMENTED
[03:59] <yessopie> (you can use that constant if you include winerr.h)
[03:59] <CategoryApl> well answer with winning a compo

[04:02] <yessopie> if ( SetCurrentDirectoryW ( L"." ) == FALSE && GetLastError () == ERROR_CALL_NOT_IMPLEMENTED ) { use ANSI }
*/

struct DirAccessWindowsPrivate {
	HANDLE h; //handle for findfirstfile
	WIN32_FIND_DATA f;
	WIN32_FIND_DATAW fu; //unicode version
};

// CreateFolderAsync

Error DirAccess::list_dir_begin(bool skip_specials) {
	_cisdir = false;
	_cishidden = false;
	_skip_specials = skip_specials;

	list_dir_end();
	p->h = FindFirstFileExW((LPCWSTR)(String(current_dir + "\\*").utf16().get_data()), FindExInfoStandard, &p->fu, FindExSearchNameMatch, NULL, 0);

	return (p->h == INVALID_HANDLE_VALUE) ? ERR_CANT_OPEN : OK;
}

String DirAccess::get_next() {
	if (p->h == INVALID_HANDLE_VALUE) {
		return "";
	}

	_cisdir = (p->fu.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
	_cishidden = (p->fu.dwFileAttributes & FILE_ATTRIBUTE_HIDDEN);

	String name = String::utf16((const char16_t *)(p->fu.cFileName));

	if (FindNextFileW(p->h, &p->fu) == 0) {
		FindClose(p->h);
		p->h = INVALID_HANDLE_VALUE;
	}

	return name;
}

bool DirAccess::current_is_dir() const {
	return _cisdir;
}

bool DirAccess::current_is_hidden() const {
	return _cishidden;
}

bool DirAccess::current_is_file() const {
	return !_cisdir;
}

bool DirAccess::current_is_special_dir() const {
	return _cisspecial;
}

void DirAccess::list_dir_end() {
	if (p->h != INVALID_HANDLE_VALUE) {
		FindClose(p->h);
		p->h = INVALID_HANDLE_VALUE;
	}
}
int DirAccess::get_drive_count() {
	return drive_count;
}
String DirAccess::get_drive(int p_drive) {
	if (p_drive < 0 || p_drive >= drive_count) {
		return "";
	}

	return String::chr(drives[p_drive]) + ":";
}

Error DirAccess::change_dir(String p_dir) {
	GLOBAL_LOCK_FUNCTION

	WCHAR real_current_dir_name[2048];
	GetCurrentDirectoryW(2048, real_current_dir_name);
	String prev_dir = String::utf16((const char16_t *)real_current_dir_name);

	SetCurrentDirectoryW((LPCWSTR)(current_dir.utf16().get_data()));
	bool worked = (SetCurrentDirectoryW((LPCWSTR)(p_dir.utf16().get_data())) != 0);

	String base;
	if (base != "") {
		GetCurrentDirectoryW(2048, real_current_dir_name);
		String new_dir = String::utf16((const char16_t *)real_current_dir_name).replace("\\", "/");
		if (!new_dir.begins_with(base)) {
			worked = false;
		}
	}

	if (worked) {
		GetCurrentDirectoryW(2048, real_current_dir_name);
		current_dir = String::utf16((const char16_t *)real_current_dir_name);
		current_dir = current_dir.replace("\\", "/");
	}

	SetCurrentDirectoryW((LPCWSTR)(prev_dir.utf16().get_data()));

	return worked ? OK : ERR_INVALID_PARAMETER;
}

Error DirAccess::make_dir(String p_dir) {
	GLOBAL_LOCK_FUNCTION

	if (p_dir.is_rel_path()) {
		p_dir = current_dir.plus_file(p_dir);
	}

	p_dir = p_dir.simplify_path().replace("/", "\\");

	bool success;
	int err;

	if (!p_dir.is_network_share_path()) {
		p_dir = "\\\\?\\" + p_dir;
		// Add "\\?\" to the path to extend max. path length past 248, if it's not a network share UNC path.
		// See https://msdn.microsoft.com/en-us/library/windows/desktop/aa363855(v=vs.85).aspx
	}

	success = CreateDirectoryW((LPCWSTR)(p_dir.utf16().get_data()), NULL);
	err = GetLastError();

	if (success) {
		return OK;
	}

	if (err == ERROR_ALREADY_EXISTS || err == ERROR_ACCESS_DENIED) {
		return ERR_ALREADY_EXISTS;
	}

	return ERR_CANT_CREATE;
}

String DirAccess::get_current_dir() {
	String base;
	if (base != "") {
		String bd = current_dir.replace("\\", "/").replace_first(base, "");
		if (bd.begins_with("/")) {
			return bd.substr(1, bd.length());
		} else {
			return bd;
		}
	}

	return current_dir;
}

String DirAccess::get_current_dir_without_drive() {
	String dir = get_current_dir();

	int p = current_dir.find(":");
	if (p != -1) {
		dir = dir.right(p + 1);
	}

	return dir;
}

bool DirAccess::file_exists(String p_file) {
	GLOBAL_LOCK_FUNCTION

	if (!p_file.is_abs_path()) {
		p_file = get_current_dir().plus_file(p_file);
	}

	DWORD fileAttr;

	fileAttr = GetFileAttributesW((LPCWSTR)(p_file.utf16().get_data()));
	if (INVALID_FILE_ATTRIBUTES == fileAttr) {
		return false;
	}

	return !(fileAttr & FILE_ATTRIBUTE_DIRECTORY);
}

bool DirAccess::dir_exists(String p_dir) {
	GLOBAL_LOCK_FUNCTION

	if (p_dir.is_rel_path()) {
		p_dir = get_current_dir().plus_file(p_dir);
	}

	DWORD fileAttr;

	fileAttr = GetFileAttributesW((LPCWSTR)(p_dir.utf16().get_data()));
	if (INVALID_FILE_ATTRIBUTES == fileAttr) {
		return false;
	}

	return (fileAttr & FILE_ATTRIBUTE_DIRECTORY);
}

Error DirAccess::rename(String p_path, String p_new_path) {
	if (p_path.is_rel_path()) {
		p_path = get_current_dir().plus_file(p_path);
	}

	if (p_new_path.is_rel_path()) {
		p_new_path = get_current_dir().plus_file(p_new_path);
	}

	// If we're only changing file name case we need to do a little juggling
	if (p_path.to_lower() == p_new_path.to_lower()) {
		if (dir_exists(p_path)) {
			// The path is a dir; just rename
			return ::_wrename((LPCWSTR)(p_path.utf16().get_data()), (LPCWSTR)(p_new_path.utf16().get_data())) == 0 ? OK : FAILED;
		}

		// The path is a file; juggle
		WCHAR tmpfile[MAX_PATH];

		if (!GetTempFileNameW((LPCWSTR)(get_current_dir().utf16().get_data()), NULL, 0, tmpfile)) {
			return FAILED;
		}

		if (!::ReplaceFileW(tmpfile, (LPCWSTR)(p_path.utf16().get_data()), NULL, 0, NULL, NULL)) {
			DeleteFileW(tmpfile);
			return FAILED;
		}

		return ::_wrename(tmpfile, (LPCWSTR)(p_new_path.utf16().get_data())) == 0 ? OK : FAILED;

	} else {
		if (file_exists(p_new_path)) {
			if (remove(p_new_path) != OK) {
				return FAILED;
			}
		}

		return ::_wrename((LPCWSTR)(p_path.utf16().get_data()), (LPCWSTR)(p_new_path.utf16().get_data())) == 0 ? OK : FAILED;
	}
}

Error DirAccess::remove(String p_path) {
	if (p_path.is_rel_path()) {
		p_path = get_current_dir().plus_file(p_path);
	}

	DWORD fileAttr;

	fileAttr = GetFileAttributesW((LPCWSTR)(p_path.utf16().get_data()));
	if (INVALID_FILE_ATTRIBUTES == fileAttr) {
		return FAILED;
	}

	if ((fileAttr & FILE_ATTRIBUTE_DIRECTORY)) {
		return ::_wrmdir((LPCWSTR)(p_path.utf16().get_data())) == 0 ? OK : FAILED;
	} else {
		return ::_wunlink((LPCWSTR)(p_path.utf16().get_data())) == 0 ? OK : FAILED;
	}
}
/*

FileType DirAccess::get_file_type(const String& p_file) const {
	WCHAR real_current_dir_name[2048];
	GetCurrentDirectoryW(2048, real_current_dir_name);
	String prev_dir = Strong::utf16((const char16_t *)real_current_dir_name);

	bool worked = SetCurrentDirectoryW((LPCWSTR)(current_dir.utf16().get_data()));

	DWORD attr;
	if (worked) {
		WIN32_FILE_ATTRIBUTE_DATA fileInfo;
		attr = GetFileAttributesExW((LPCWSTR)(p_file.utf16().get_data()), GetFileExInfoStandard, &fileInfo);
	}

	SetCurrentDirectoryW((LPCWSTR)(prev_dir.utf16().get_data()));

	if (!worked) {
		return FILE_TYPE_NONE;
	}

	return (attr & FILE_ATTRIBUTE_DIRECTORY) ? FILE_TYPE_
}
*/

uint64_t DirAccess::get_space_left() {
	uint64_t bytes = 0;

	if (!GetDiskFreeSpaceEx(NULL, (PULARGE_INTEGER)&bytes, NULL, NULL)) {
		return 0;
	}

	//this is either 0 or a value in bytes.
	return bytes;
}

String DirAccess::get_filesystem_type() const {
	String path = const_cast<DirAccess *>(this)->get_current_dir();

	if (path.is_network_share_path()) {
		return "Network Share";
	}

	int unit_end = path.find(":");
	ERR_FAIL_COND_V(unit_end == -1, String());
	String unit = path.substr(0, unit_end + 1) + "\\";

	WCHAR szVolumeName[100];
	WCHAR szFileSystemName[10];
	DWORD dwSerialNumber = 0;
	DWORD dwMaxFileNameLength = 0;
	DWORD dwFileSystemFlags = 0;

	if (::GetVolumeInformationW((LPCWSTR)(unit.utf16().get_data()),
				szVolumeName,
				sizeof(szVolumeName),
				&dwSerialNumber,
				&dwMaxFileNameLength,
				&dwFileSystemFlags,
				szFileSystemName,
				sizeof(szFileSystemName)) == TRUE) {
		return String::utf16((const char16_t *)szFileSystemName);
	}

	ERR_FAIL_V("");
}

int DirAccess::get_current_drive() {
	String path = get_current_dir().to_lower();
	for (int i = 0; i < get_drive_count(); i++) {
		String d = get_drive(i).to_lower();
		if (path.begins_with(d)) {
			return i;
		}
	}

	return 0;
}

bool DirAccess::drives_are_shortcuts() {
	return false;
}

uint64_t DirAccess::get_modified_time(String p_file) {
	return 0;
};

DirAccess::DirAccess() {
	p = memnew(DirAccessWindowsPrivate);
	p->h = INVALID_HANDLE_VALUE;
	current_dir = ".";

	drive_count = 0;

#ifdef UWP_ENABLED
	Windows::Storage::StorageFolder ^ install_folder = Windows::ApplicationModel::Package::Current->InstalledLocation;
	change_dir(install_folder->Path->Data());

#else

	DWORD mask = GetLogicalDrives();

	for (int i = 0; i < MAX_DRIVES; i++) {
		if (mask & (1 << i)) { //DRIVE EXISTS

			drives[drive_count] = 'A' + i;
			drive_count++;
		}
	}

	change_dir(".");
#endif
}

DirAccess::~DirAccess() {
	list_dir_end();

	memdelete(p);
}

#else

Error DirAccess::list_dir_begin(bool skip_specials) {
	list_dir_end(); //close any previous dir opening!

	_skip_specials = skip_specials;

	//char real_current_dir_name[2048]; //is this enough?!
	//getcwd(real_current_dir_name,2048);
	//chdir(current_path.utf8().get_data());
	dir_stream = opendir(current_dir.utf8().get_data());
	//chdir(real_current_dir_name);
	if (!dir_stream) {
		return ERR_CANT_OPEN; //error!
	}

	return OK;
}

bool DirAccess::file_exists(String p_file) {
	GLOBAL_LOCK_FUNCTION

	if (p_file.is_rel_path()) {
		p_file = current_dir.plus_file(p_file);
	}

	struct stat flags;
	bool success = (stat(p_file.utf8().get_data(), &flags) == 0);

	if (success && S_ISDIR(flags.st_mode)) {
		success = false;
	}

	return success;
}

bool DirAccess::dir_exists(String p_dir) {
	GLOBAL_LOCK_FUNCTION

	if (p_dir.is_rel_path()) {
		p_dir = get_current_dir().plus_file(p_dir);
	}

	struct stat flags;
	bool success = (stat(p_dir.utf8().get_data(), &flags) == 0);

	return (success && S_ISDIR(flags.st_mode));
}

uint64_t DirAccess::get_modified_time(String p_file) {
	if (p_file.is_rel_path()) {
		p_file = current_dir.plus_file(p_file);
	}

	struct stat flags;
	bool success = (stat(p_file.utf8().get_data(), &flags) == 0);

	if (success) {
		return flags.st_mtime;
	} else {
		ERR_FAIL_V(0);
	};
	return 0;
};

String DirAccess::get_next() {
	if (!dir_stream) {
		return "";
	}

	dirent *entry = readdir(dir_stream);

	if (entry == nullptr) {
		list_dir_end();
		return "";
	}

	String fname = fix_unicode_name(entry->d_name);

	// Look at d_type to determine if the entry is a directory, unless
	// its type is unknown (the file system does not support it) or if
	// the type is a link, in that case we want to resolve the link to
	// known if it points to a directory. stat() will resolve the link
	// for us.
	if (entry->d_type == DT_UNKNOWN || entry->d_type == DT_LNK) {
		String f = current_dir.plus_file(fname);

		struct stat flags;
		if (stat(f.utf8().get_data(), &flags) == 0) {
			_cisdir = S_ISDIR(flags.st_mode);
		} else {
			_cisdir = false;
		}
	} else {
		_cisdir = (entry->d_type == DT_DIR);
	}

	_cishidden = is_hidden(fname);

	_cisspecial = is_special(fname);

	if (_skip_specials && _cisspecial) {
		// Should only happen 2 times max
		return get_next();
	}

	return fname;
}

bool DirAccess::current_is_dir() const {
	return _cisdir;
}

bool DirAccess::current_is_file() const {
	return !_cisdir;
}

bool DirAccess::current_is_special_dir() const {
	return _cisspecial;
}

bool DirAccess::current_is_hidden() const {
	return _cishidden;
}

void DirAccess::list_dir_end() {
	if (dir_stream) {
		closedir(dir_stream);
	}
	dir_stream = nullptr;
	_cisdir = false;
}

#if defined(HAVE_MNTENT) && defined(X11_ENABLED)
static bool _filter_drive(struct mntent *mnt) {
	// Ignore devices that don't point to /dev
	if (strncmp(mnt->mnt_fsname, "/dev", 4) != 0) {
		return false;
	}

	// Accept devices mounted at common locations
	if (strncmp(mnt->mnt_dir, "/media", 6) == 0 ||
			strncmp(mnt->mnt_dir, "/mnt", 4) == 0 ||
			strncmp(mnt->mnt_dir, "/home", 5) == 0 ||
			strncmp(mnt->mnt_dir, "/run/media", 10) == 0) {
		return true;
	}

	// Ignore everything else
	return false;
}
#endif

static void _get_drives(List<String> *list) {
	list->push_back("/");

#if defined(HAVE_MNTENT) && defined(X11_ENABLED)
	// Check /etc/mtab for the list of mounted partitions
	FILE *mtab = setmntent("/etc/mtab", "r");
	if (mtab) {
		struct mntent mnt;
		char strings[4096];

		while (getmntent_r(mtab, &mnt, strings, sizeof(strings))) {
			if (mnt.mnt_dir != nullptr && _filter_drive(&mnt)) {
				// Avoid duplicates
				if (!list->find(mnt.mnt_dir)) {
					list->push_back(mnt.mnt_dir);
				}
			}
		}

		endmntent(mtab);
	}
#endif

	// Add $HOME
	const char *home = getenv("HOME");
	if (home) {
		// Only add if it's not a duplicate
		if (!list->find(home)) {
			list->push_back(home);
		}

		// Check $HOME/.config/gtk-3.0/bookmarks
		char path[1024];
		snprintf(path, 1024, "%s/.config/gtk-3.0/bookmarks", home);
		FILE *fd = fopen(path, "r");
		if (fd) {
			char string[1024];
			while (fgets(string, 1024, fd)) {
				// Parse only file:// links
				if (strncmp(string, "file://", 7) == 0) {
					// Strip any unwanted edges on the strings and push_back if it's not a duplicate
					String fpath = String(string + 7).strip_edges().split_spaces()[0].percent_decode();
					if (!list->find(fpath)) {
						list->push_back(fpath);
					}
				}
			}

			fclose(fd);
		}
	}

	list->sort();
}

int DirAccess::get_drive_count() {
	List<String> list;
	_get_drives(&list);

	return list.size();
}

String DirAccess::get_drive(int p_drive) {
	List<String> list;
	_get_drives(&list);

	ERR_FAIL_INDEX_V(p_drive, list.size(), "");

	return list[p_drive];
}

int DirAccess::get_current_drive() {
	int drive = 0;
	int max_length = -1;
	const String path = get_current_dir().to_lower();
	for (int i = 0; i < get_drive_count(); i++) {
		const String d = get_drive(i).to_lower();
		if (max_length < d.length() && path.begins_with(d)) {
			max_length = d.length();
			drive = i;
		}
	}
	return drive;
}

bool DirAccess::drives_are_shortcuts() {
	return true;
}

Error DirAccess::make_dir(String p_dir) {
	GLOBAL_LOCK_FUNCTION

	if (p_dir.is_rel_path()) {
		p_dir = get_current_dir().plus_file(p_dir);
	}

	bool success = (mkdir(p_dir.utf8().get_data(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0);
	int err = errno;

	if (success) {
		return OK;
	};

	if (err == EEXIST) {
		return ERR_ALREADY_EXISTS;
	};

	return ERR_CANT_CREATE;
}

Error DirAccess::change_dir(String p_dir) {
	GLOBAL_LOCK_FUNCTION

	// prev_dir is the directory we are changing out of
	String prev_dir;
	char real_current_dir_name[2048];
	ERR_FAIL_COND_V(getcwd(real_current_dir_name, 2048) == nullptr, ERR_BUG);
	if (prev_dir.parse_utf8(real_current_dir_name)) {
		prev_dir = real_current_dir_name; //no utf8, maybe latin?
	}

	// try_dir is the directory we are trying to change into
	String try_dir = "";
	if (p_dir.is_rel_path()) {
		String next_dir = current_dir.plus_file(p_dir);
		next_dir = next_dir.simplify_path();
		try_dir = next_dir;
	} else {
		try_dir = p_dir;
	}

	bool worked = (chdir(try_dir.utf8().get_data()) == 0); // we can only give this utf8
	if (!worked) {
		return ERR_INVALID_PARAMETER;
	}

	String base;
	if (base != String() && !try_dir.begins_with(base)) {
		ERR_FAIL_COND_V(getcwd(real_current_dir_name, 2048) == nullptr, ERR_BUG);
		String new_dir;
		new_dir.parse_utf8(real_current_dir_name);

		if (!new_dir.begins_with(base)) {
			try_dir = current_dir; //revert
		}
	}

	// the directory exists, so set current_dir to try_dir
	current_dir = try_dir;
	ERR_FAIL_COND_V(chdir(prev_dir.utf8().get_data()) != 0, ERR_BUG);
	return OK;
}

String DirAccess::get_current_dir() {
	String base;
	if (base != "") {
		String bd = current_dir.replace_first(base, "");
		if (bd.begins_with("/")) {
			return bd.substr(1, bd.length());
		} else {
			return bd;
		}
	}
	return current_dir;
}

Error DirAccess::rename(String p_path, String p_new_path) {
	if (p_path.is_rel_path()) {
		p_path = get_current_dir().plus_file(p_path);
	}

	if (p_new_path.is_rel_path()) {
		p_new_path = get_current_dir().plus_file(p_new_path);
	}

	return ::rename(p_path.utf8().get_data(), p_new_path.utf8().get_data()) == 0 ? OK : FAILED;
}

Error DirAccess::remove(String p_path) {
	if (p_path.is_rel_path()) {
		p_path = get_current_dir().plus_file(p_path);
	}

	struct stat flags;
	if ((stat(p_path.utf8().get_data(), &flags) != 0)) {
		return FAILED;
	}

	if (S_ISDIR(flags.st_mode)) {
		return ::rmdir(p_path.utf8().get_data()) == 0 ? OK : FAILED;
	} else {
		return ::unlink(p_path.utf8().get_data()) == 0 ? OK : FAILED;
	}
}

bool DirAccess::is_link(String p_file) {
	if (p_file.is_rel_path()) {
		p_file = get_current_dir().plus_file(p_file);
	}

	struct stat flags;
	if ((lstat(p_file.utf8().get_data(), &flags) != 0)) {
		return FAILED;
	}

	return S_ISLNK(flags.st_mode);
}

String DirAccess::read_link(String p_file) {
	if (p_file.is_rel_path()) {
		p_file = get_current_dir().plus_file(p_file);
	}

	char buf[256];
	memset(buf, 0, 256);
	ssize_t len = readlink(p_file.utf8().get_data(), buf, sizeof(buf));
	String link;
	if (len > 0) {
		link.parse_utf8(buf, len);
	}
	return link;
}

Error DirAccess::create_link(String p_source, String p_target) {
	if (p_target.is_rel_path()) {
		p_target = get_current_dir().plus_file(p_target);
	}

	if (symlink(p_source.utf8().get_data(), p_target.utf8().get_data()) == 0) {
		return OK;
	} else {
		return FAILED;
	}
}

uint64_t DirAccess::get_space_left() {
#ifndef NO_STATVFS
	struct statvfs vfs;
	if (statvfs(current_dir.utf8().get_data(), &vfs) != 0) {
		return 0;
	};

	return (uint64_t)vfs.f_bavail * (uint64_t)vfs.f_frsize;
#else
	// FIXME: Implement this.
	return 0;
#endif
};

String DirAccess::get_filesystem_type() const {
	return ""; //TODO this should be implemented
}

bool DirAccess::is_hidden(const String &p_name) {
	return p_name != "." && p_name != ".." && p_name.begins_with(".");
}

String DirAccess::get_current_dir_without_drive() {
	return get_current_dir();
}

DirAccess::DirAccess() {
	dir_stream = NULL;
	_cisdir = false;

	next_is_dir = false;
	_skip_specials = false;

	_cishidden = false;
	_cisspecial = false;

	/* determine drive count */

	// set current directory to an absolute path of the current directory
	char real_current_dir_name[2048];
	ERR_FAIL_COND(getcwd(real_current_dir_name, 2048) == nullptr);
	if (current_dir.parse_utf8(real_current_dir_name)) {
		current_dir = real_current_dir_name;
	}

	change_dir(current_dir);
}

DirAccess::~DirAccess() {
	list_dir_end();
}

#endif

static Error _erase_recursive(DirAccess *da) {
	List<String> dirs;
	List<String> files;

	da->list_dir_begin();
	String n = da->get_next();
	while (n != String()) {
		if (n != "." && n != "..") {
			if (da->current_is_dir()) {
				dirs.push_back(n);
			} else {
				files.push_back(n);
			}
		}

		n = da->get_next();
	}

	da->list_dir_end();

	for (List<String>::Element *E = dirs.front(); E; E = E->next()) {
		Error err = da->change_dir(E->get());
		if (err == OK) {
			err = _erase_recursive(da);
			if (err) {
				da->change_dir("..");
				return err;
			}
			err = da->change_dir("..");
			if (err) {
				return err;
			}
			err = da->remove(da->get_current_dir().plus_file(E->get()));
			if (err) {
				return err;
			}
		} else {
			return err;
		}
	}

	for (List<String>::Element *E = files.front(); E; E = E->next()) {
		Error err = da->remove(da->get_current_dir().plus_file(E->get()));
		if (err) {
			return err;
		}
	}

	return OK;
}

Error DirAccess::erase_contents_recursive() {
	return _erase_recursive(this);
}

Error DirAccess::make_dir_recursive(String p_dir) {
	if (p_dir.length() < 1) {
		return OK;
	};

	String full_dir;

	if (p_dir.is_rel_path()) {
		//append current
		full_dir = get_current_dir().plus_file(p_dir);

	} else {
		full_dir = p_dir;
	}

	full_dir = full_dir.replace("\\", "/");

	String base;

	if (full_dir.is_network_share_path()) {
		int pos = full_dir.find("/", 2);
		ERR_FAIL_COND_V(pos < 0, ERR_INVALID_PARAMETER);
		pos = full_dir.find("/", pos + 1);
		ERR_FAIL_COND_V(pos < 0, ERR_INVALID_PARAMETER);
		base = full_dir.substr(0, pos + 1);
	} else if (full_dir.begins_with("/")) {
		base = "/";
	} else if (full_dir.find(":/") != -1) {
		base = full_dir.substr(0, full_dir.find(":/") + 2);
	} else {
		ERR_FAIL_V(ERR_INVALID_PARAMETER);
	}

	full_dir = full_dir.replace_first(base, "").simplify_path();

	Vector<String> subdirs = full_dir.split("/");

	String curpath = base;
	for (int i = 0; i < subdirs.size(); i++) {
		curpath = curpath.plus_file(subdirs[i]);
		Error err = make_dir(curpath);
		if (err != OK && err != ERR_ALREADY_EXISTS) {
			ERR_FAIL_V_MSG(err, "Could not create directory: " + curpath);
		}
	}

	return OK;
}

DirAccess *DirAccess::create_for_path(const String &p_path) {
	DirAccess *d = memnew(DirAccess());
	d->open(p_path);
	return d;
}
DirAccess *DirAccess::create() {
	return memnew(DirAccess());
}

Error DirAccess::open(const String &p_path) {
	return change_dir(p_path);
}

String DirAccess::get_full_path(const String &p_path) {
	DirAccess d;

	d.change_dir(p_path);
	String full = d.get_current_dir();

	return full;
}

Error DirAccess::copy(String p_from, String p_to, int p_chmod_flags) {
	//printf("copy %s -> %s\n",p_from.ascii().get_data(),p_to.ascii().get_data());
	Error err;
	FileAccess *fsrc = FileAccess::create_and_open(p_from, FileAccess::READ, &err);

	if (err) {
		ERR_PRINT("Failed to open " + p_from);
		return err;
	}

	FileAccess *fdst = FileAccess::create_and_open(p_to, FileAccess::WRITE, &err);
	if (err) {
		fsrc->close();
		memdelete(fsrc);
		ERR_PRINT("Failed to open " + p_to);
		return err;
	}

	const size_t copy_buffer_limit = 65536; // 64 KB

	fsrc->seek_end(0);
	uint64_t size = fsrc->get_position();
	fsrc->seek(0);
	err = OK;
	size_t buffer_size = MIN(size * sizeof(uint8_t), copy_buffer_limit);
	LocalVector<uint8_t> buffer;
	buffer.resize(buffer_size);
	while (size > 0) {
		if (fsrc->get_error() != OK) {
			err = fsrc->get_error();
			break;
		}
		if (fdst->get_error() != OK) {
			err = fdst->get_error();
			break;
		}

		int bytes_read = fsrc->get_buffer(buffer.ptr(), buffer_size);
		if (bytes_read <= 0) {
			err = FAILED;
			break;
		}
		fdst->store_buffer(buffer.ptr(), bytes_read);

		size -= bytes_read;
	}

	if (err == OK && p_chmod_flags != -1) {
		fdst->close();
		err = FileAccess::set_unix_permissions(p_to, p_chmod_flags);
		// If running on a platform with no chmod support (i.e., Windows), don't fail
		if (err == ERR_UNAVAILABLE) {
			err = OK;
		}
	}

	memdelete(fsrc);
	memdelete(fdst);

	return err;
}

// Changes dir for the current scope, returning back to the original dir
// when scope exits
class DirChanger {
	DirAccess *da;
	String original_dir;

public:
	DirChanger(DirAccess *p_da, String p_dir) :
			da(p_da),
			original_dir(p_da->get_current_dir()) {
		p_da->change_dir(p_dir);
	}

	~DirChanger() {
		da->change_dir(original_dir);
	}
};

Error DirAccess::_copy_dir(DirAccess *p_target_da, String p_to, int p_chmod_flags, bool p_copy_links) {
	List<String> dirs;

	String curdir = get_current_dir();
	list_dir_begin();
	String n = get_next();
	while (n != String()) {
		if (n != "." && n != "..") {
			if (p_copy_links && is_link(get_current_dir().plus_file(n))) {
				create_link(read_link(get_current_dir().plus_file(n)), p_to + n);
			} else if (current_is_dir()) {
				dirs.push_back(n);
			} else {
				const String &rel_path = n;
				if (!n.is_rel_path()) {
					list_dir_end();
					return ERR_BUG;
				}
				Error err = copy(get_current_dir().plus_file(n), p_to + rel_path, p_chmod_flags);
				if (err) {
					list_dir_end();
					return err;
				}
			}
		}

		n = get_next();
	}

	list_dir_end();

	for (List<String>::Element *E = dirs.front(); E; E = E->next()) {
		String rel_path = E->get();
		String target_dir = p_to + rel_path;
		if (!p_target_da->dir_exists(target_dir)) {
			Error err = p_target_da->make_dir(target_dir);
			ERR_FAIL_COND_V_MSG(err != OK, err, "Cannot create directory '" + target_dir + "'.");
		}

		Error err = change_dir(E->get());
		ERR_FAIL_COND_V_MSG(err != OK, err, "Cannot change current directory to '" + E->get() + "'.");

		err = _copy_dir(p_target_da, p_to + rel_path + "/", p_chmod_flags, p_copy_links);
		if (err) {
			change_dir("..");
			ERR_FAIL_V_MSG(err, "Failed to copy recursively.");
		}
		err = change_dir("..");
		ERR_FAIL_COND_V_MSG(err != OK, err, "Failed to go back.");
	}

	return OK;
}

Error DirAccess::copy_dir(String p_from, String p_to, int p_chmod_flags, bool p_copy_links) {
	ERR_FAIL_COND_V_MSG(!dir_exists(p_from), ERR_FILE_NOT_FOUND, "Source directory doesn't exist.");

	DirAccess *target_da = DirAccess::create_for_path(p_to);
	ERR_FAIL_COND_V_MSG(!target_da, ERR_CANT_CREATE, "Cannot create DirAccess for path '" + p_to + "'.");

	if (!target_da->dir_exists(p_to)) {
		Error err = target_da->make_dir_recursive(p_to);
		if (err) {
			memdelete(target_da);
		}
		ERR_FAIL_COND_V_MSG(err != OK, err, "Cannot create directory '" + p_to + "'.");
	}

	if (!p_to.ends_with("/")) {
		p_to = p_to + "/";
	}

	DirChanger dir_changer(this, p_from);
	Error err = _copy_dir(target_da, p_to, p_chmod_flags, p_copy_links);
	memdelete(target_da);

	return err;
}

bool DirAccess::exists(String p_dir) {
	DirAccess *da = DirAccess::create_for_path(p_dir);
	bool valid = da->change_dir(p_dir) == OK;
	memdelete(da);
	return valid;
}

String DirAccess::get_filesystem_abspath_for(String p_path) {
	DirAccess d;

	d.change_dir(p_path);
	String full = d.get_current_dir();

	return full;
}

bool DirAccess::is_special(const String &p_path) {
	if (p_path.size() > 2) {
		return false;
	}

	return p_path == "." || p_path == "..";
}
#line 0

#line 1 "sfw/core/inet_address.cpp"
//Based on:

// Copyright 2010, Shuo Chen.  All rights reserved.
// http://code.google.com/p/muduo/
//
// Use of this source code is governed by a BSD-style license
// that can be found in the License file.

// Author: Shuo Chen (chenshuo at chenshuo dot com)

#include <cstdio>
#include <cstring>

#if defined(_WIN64) || defined(_WIN32)

#ifdef __GNUC__
#define GCCWIN
// Mingw / gcc on windows
//   #define _WIN32_WINNT 0x0501
#include <winsock2.h>
#include <ws2tcpip.h>

extern "C" {
WINSOCK_API_LINKAGE INT WSAAPI inet_pton(INT Family, PCSTR pszAddrString, PVOID pAddrBuf);

#ifdef __MINGW64_VERSION_MAJOR
#if __MINGW64_VERSION_MAJOR >= 7
WINSOCK_API_LINKAGE PCSTR WSAAPI inet_ntop(INT Family, const VOID *pAddr, PSTR pStringBuf, size_t StringBufSize);
#else
WINSOCK_API_LINKAGE PCSTR WSAAPI inet_ntop(INT Family, VOID *pAddr, PSTR pStringBuf, size_t StringBufSize);
#endif

#else
WINSOCK_API_LINKAGE PCSTR WSAAPI inet_ntop(INT Family, VOID *pAddr, PSTR pStringBuf, size_t StringBufSize);
#endif
}
#else
// Windows...

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <in6addr.h>
#include <winsock2.h>
#include <ws2tcpip.h>

extern "C" {

PCSTR WSAAPI inet_ntop(_In_ INT Family, _In_ const VOID *pAddr, _Out_writes_(StringBufSize) PSTR pStringBuf, _In_ size_t StringBufSize);

WINSOCK_API_LINKAGE INT WSAAPI inet_pton(_In_ INT Family, _In_ PCSTR pszAddrString, _When_(Family == AF_INET, _Out_writes_bytes_(sizeof(IN_ADDR))) _When_(Family == AF_INET6, _Out_writes_bytes_(sizeof(IN6_ADDR))) PVOID pAddrBuf);
}

#endif

struct in6__addruint {
	union {
		u_char Byte[16];
		u_short Word[8];
		uint32_t __s6_addr32[4];
	} uext;
};
#else
#include <netdb.h>
#include <netinet/tcp.h>
#include <strings.h> // memset
#endif

#ifdef _MSC_VER
#pragma comment(lib, "ws2_32")
#endif

// INADDR_ANY use (type)value casting.
static const in_addr_t kInaddrAny = INADDR_ANY;
static const in_addr_t kInaddrLoopback = INADDR_ANY;

//     /* Structure describing an Internet socket address.  */
//     struct sock_addrin {
//         sa_family_t    sin_family; /* address family: AF_INET */
//         uint16_t       sin_port;   /* port in network byte order */
//         struct in_addr sin_addr;   /* internet address */
//     };

//     /* Internet address. */
//     typedef uint32_t in_addr_t;
//     struct in_addr {
//         in_addr_t       s_addr;     /* address in network byte order */
//     };

//     struct sockaddr_in6 {
//         sa_family_t     sin6_family;   /* address family: AF_INET6 */
//         uint16_t        sin6_port;     /* port in network byte order */
//         uint32_t        sin6_flowinfo; /* IPv6 flow information */
//         struct in6_addr sin6_addr;     /* IPv6 address */
//         uint32_t        sin6_scope_id; /* IPv6 scope-id */
//     };

/*
#ifdef __linux__
#if !(__GNUC_PREREQ(4, 6))
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#endif
#endif
*/

// Defined here so we don't need to include <windows.h> in the header
struct InetAddress::InetAddressData {
	union {
		struct sockaddr_in _addr;
		struct sockaddr_in6 _addr6;
	};
};

String InetAddress::to_ip_port() const {
	char buf[64] = "";
	uint16_t port = ntohs(_data->_addr.sin_port);
	snprintf(buf, sizeof(buf), ":%u", port);

	return to_ip() + String(buf);
}

bool InetAddress::is_intranet_ip() const {
	if (_data->_addr.sin_family == AF_INET) {
		uint32_t ip_addr = ntohl(_data->_addr.sin_addr.s_addr);
		if ((ip_addr >= 0x0A000000 && ip_addr <= 0x0AFFFFFF) ||
				(ip_addr >= 0xAC100000 && ip_addr <= 0xAC1FFFFF) ||
				(ip_addr >= 0xC0A80000 && ip_addr <= 0xC0A8FFFF) ||
				ip_addr == 0x7f000001)

		{
			return true;
		}
	} else {
		auto addrP = ip6_net_endian();
		// Loopback ip
		if (*addrP == 0 && *(addrP + 1) == 0 && *(addrP + 2) == 0 &&
				ntohl(*(addrP + 3)) == 1) {
			return true;
		}

		// Privated ip is prefixed by FEC0::/10 or FE80::/10, need testing
		auto i32 = (ntohl(*addrP) & 0xffc00000);
		if (i32 == 0xfec00000 || i32 == 0xfe800000) {
			return true;
		}

		if (*addrP == 0 && *(addrP + 1) == 0 && ntohl(*(addrP + 2)) == 0xffff) {
			// the IPv6 version of an IPv4 IP address
			uint32_t ip_addr = ntohl(*(addrP + 3));
			if ((ip_addr >= 0x0A000000 && ip_addr <= 0x0AFFFFFF) ||
					(ip_addr >= 0xAC100000 && ip_addr <= 0xAC1FFFFF) ||
					(ip_addr >= 0xC0A80000 && ip_addr <= 0xC0A8FFFF) ||
					ip_addr == 0x7f000001)

			{
				return true;
			}
		}
	}
	return false;
}

bool InetAddress::is_loopback_ip() const {
	if (!is_ip_v6()) {
		uint32_t ip_addr = ntohl(_data->_addr.sin_addr.s_addr);
		if (ip_addr == 0x7f000001) {
			return true;
		}
	} else {
		auto addrP = ip6_net_endian();
		if (*addrP == 0 && *(addrP + 1) == 0 && *(addrP + 2) == 0 &&
				ntohl(*(addrP + 3)) == 1) {
			return true;
		}
		// the IPv6 version of an IPv4 loopback address
		if (*addrP == 0 && *(addrP + 1) == 0 && ntohl(*(addrP + 2)) == 0xffff &&
				ntohl(*(addrP + 3)) == 0x7f000001) {
			return true;
		}
	}
	return false;
}

const struct sockaddr *InetAddress::get_sock_addr() const {
	return static_cast<const struct sockaddr *>((void *)(&_data->_addr6));
}

void InetAddress::set_sock_addr_inet6(const struct sockaddr_in6 &addr6) {
	_data->_addr6 = addr6;
	_is_ip_v6 = (_data->_addr6.sin6_family == AF_INET6);
	_is_unspecified = false;
}

sa_family_t InetAddress::family() const {
	return _data->_addr.sin_family;
}

String InetAddress::to_ip() const {
	char buf[64];
	if (_data->_addr.sin_family == AF_INET) {
#if defined GCCWIN || (_MSC_VER && _MSC_VER >= 1900)
		::inet_ntop(AF_INET, (PVOID)&_data->_addr.sin_addr, buf, sizeof(buf));
#else
		::inet_ntop(AF_INET, &_data->_addr.sin_addr, buf, sizeof(buf));
#endif
	} else if (_data->_addr.sin_family == AF_INET6) {
#if defined GCCWIN || (_MSC_VER && _MSC_VER >= 1900)
		::inet_ntop(AF_INET6, (PVOID)&_data->_addr6.sin6_addr, buf, sizeof(buf));
#else
		::inet_ntop(AF_INET6, &_data->_addr6.sin6_addr, buf, sizeof(buf));
#endif
	}

	return buf;
}

uint32_t InetAddress::ip_net_endian() const {
	// assert(family() == AF_INET);
	return _data->_addr.sin_addr.s_addr;
}

const uint32_t *InetAddress::ip6_net_endian() const {
// assert(family() == AF_INET6);
#if defined __linux__ || defined __HAIKU__
	return _data->_addr6.sin6_addr.s6_addr32;
#elif defined(_WIN64) || defined(_WIN32)
	// TODO is this OK ?
	const struct in6__addruint *_addrtemp =
			reinterpret_cast<const struct in6__addruint *>(&_data->_addr6.sin6_addr);
	return (*_addrtemp).uext.__s6_addr32;
#else
	return _data->_addr6.sin6_addr.__u6_addr.__u6_addr32;
#endif
}

uint16_t InetAddress::port_net_endian() const {
	return _data->_addr.sin_port;
}

void InetAddress::set_port_net_endian(uint16_t port) {
	_data->_addr.sin_port = port;
}

inline bool InetAddress::is_unspecified() const {
	return _is_unspecified;
}

uint16_t InetAddress::to_port() const {
	return ntohs(port_net_endian());
}

bool InetAddress::is_ip_v6() const {
	return _is_ip_v6;
}

InetAddress::InetAddress(uint16_t port, bool loopbackOnly, bool ipv6) {
	_data = memnew(InetAddressData);

	_is_ip_v6 = ipv6;

	if (ipv6) {
		memset(&_data->_addr6, 0, sizeof(_data->_addr6));
		_data->_addr6.sin6_family = AF_INET6;

		in6_addr ip = loopbackOnly ? in6addr_loopback : in6addr_any;

		_data->_addr6.sin6_addr = ip;
		_data->_addr6.sin6_port = htons(port);
	} else {
		memset(&_data->_addr, 0, sizeof(_data->_addr));
		_data->_addr.sin_family = AF_INET;

		in_addr_t ip = loopbackOnly ? kInaddrLoopback : kInaddrAny;

		_data->_addr.sin_addr.s_addr = htonl(ip);
		_data->_addr.sin_port = htons(port);
	}

	_is_unspecified = false;
}

InetAddress::InetAddress(const String &ip, uint16_t port, bool ipv6) {
	_data = memnew(InetAddressData);

	_is_ip_v6 = ipv6;

	if (ipv6) {
		memset(&_data->_addr6, 0, sizeof(_data->_addr6));
		_data->_addr6.sin6_family = AF_INET6;
		_data->_addr6.sin6_port = htons(port);

		if (::inet_pton(AF_INET6, ip.utf8().get_data(), &_data->_addr6.sin6_addr) <= 0) {
			return;
		}
	} else {
		memset(&_data->_addr, 0, sizeof(_data->_addr));
		_data->_addr.sin_family = AF_INET;
		_data->_addr.sin_port = htons(port);

		if (::inet_pton(AF_INET, ip.utf8().get_data(), &_data->_addr.sin_addr) <= 0) {
			return;
		}
	}

	_is_unspecified = false;
}

InetAddress::InetAddress(const struct sockaddr_in &addr) {
	_data = memnew(InetAddressData);

	_data->_addr = addr;
	_is_unspecified = false;
}

InetAddress::InetAddress(const struct sockaddr_in6 &addr) {
	_data = memnew(InetAddressData);

	_data->_addr6 = addr;
	_is_ip_v6 = true;
	_is_unspecified = false;
}

InetAddress::~InetAddress() {
	memdelete(_data);
}
#line 0

#line 1 "sfw/core/socket.cpp"

//Based on:

/**
 *
 *  Socket.cc
 *  An Tao
 *
 *  Public header file in trantor lib.
 *
 *  Copyright 2018, An Tao.  All rights reserved.
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the License file.
 *
 *
 */

#if !defined(_WIN64) && !defined(_WIN32)
#include <unistd.h>
#endif
#include <fcntl.h>

///usr/include/asm-generic/errno-base.h
//http://www.virtsync.com/c-error-codes-include-errno
#include <cerrno>

#if defined(_WIN64) || defined(_WIN32)
#include <ws2tcpip.h>
#else
#include <netinet/tcp.h>
#include <sys/socket.h>
#endif

void Socket::create_net_socket() {
	create(AF_INET);
}

void Socket::create(int family) {
#ifdef __linux__
	_socket = ::socket(family, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, IPPROTO_TCP);
#else
	_socket = static_cast<int>(::socket(family, SOCK_STREAM, IPPROTO_TCP));
#endif
}

void Socket::close_socket() {
	if (!_socket) {
		return;
	}

#if !defined(_WIN64) && !defined(_WIN32)
	close(_socket);
#else
	closesocket(_socket);
#endif

	_socket = 0;
}

// taken from muduo
int Socket::set_non_block() {
	ERR_FAIL_COND_V(_socket == 0, -1);

#if defined(_WIN64) || defined(_WIN32)
	// TODO how to set FD_CLOEXEC on windows? is it necessary?
	u_long arg = 1;
	auto ret = ioctlsocket(_socket, (long)FIONBIO, &arg);

	if (ret) {
		//LOG_ERR("ioctlsocket error");
		return -1;
	}

	return 0;
#else
	// non-block
	int flags = ::fcntl(_socket, F_GETFL, 0);
	flags |= O_NONBLOCK;
	int ret = ::fcntl(_socket, F_SETFL, flags);
	// TODO check

	return ret;
#endif
}

// taken from muduo
int Socket::set_close_on_exit() {
	ERR_FAIL_COND_V(_socket == 0, -1);

#if defined(_WIN64) || defined(_WIN32)
	// TODO how to set FD_CLOEXEC on windows? is it necessary?
	return 0;
#else
	// close-on-exec
	int flags = ::fcntl(_socket, F_GETFD, 0);
	flags |= FD_CLOEXEC;
	int ret = ::fcntl(_socket, F_SETFD, flags);
	// TODO check

	return ret;
#endif
}

int Socket::get_error() {
	ERR_FAIL_COND_V(_socket == 0, -1);

	int optval;

	socklen_t optlen = static_cast<socklen_t>(sizeof optval);

#if defined(_WIN64) || defined(_WIN32)
	if (::getsockopt(_socket, SOL_SOCKET, SO_ERROR, (char *)&optval, &optlen) < 0)
#else
	if (::getsockopt(_socket, SOL_SOCKET, SO_ERROR, &optval, &optlen) < 0)
#endif
	{
		return errno;
	} else {
		return optval;
	}
}

int Socket::connect(const InetAddress &addr) {
	ERR_FAIL_COND_V(_socket == 0, -1);

	if (addr.is_ip_v6()) {
		return ::connect(_socket, addr.get_sock_addr(), static_cast<socklen_t>(sizeof(struct sockaddr_in6)));
	} else {
		return ::connect(_socket, addr.get_sock_addr(), static_cast<socklen_t>(sizeof(struct sockaddr_in)));
	}
}

bool Socket::is_self_connect() {
	ERR_FAIL_COND_V(_socket == 0, false);

	struct sockaddr_in6 localaddr = get_local_addr();
	struct sockaddr_in6 peeraddr = get_peer_addr();

	if (localaddr.sin6_family == AF_INET) {
		const struct sockaddr_in *laddr4 = reinterpret_cast<struct sockaddr_in *>(&localaddr);
		const struct sockaddr_in *raddr4 = reinterpret_cast<struct sockaddr_in *>(&peeraddr);
		return laddr4->sin_port == raddr4->sin_port && laddr4->sin_addr.s_addr == raddr4->sin_addr.s_addr;
	} else if (localaddr.sin6_family == AF_INET6) {
		return localaddr.sin6_port == peeraddr.sin6_port && memcmp(&localaddr.sin6_addr, &peeraddr.sin6_addr, sizeof localaddr.sin6_addr) == 0;
	} else {
		return false;
	}
}

int Socket::bind_address(const InetAddress &address) {
	ERR_FAIL_COND_V(_socket == 0, -1);

	int ret;
	if (address.is_ip_v6()) {
		ret = ::bind(_socket, address.get_sock_addr(), sizeof(sockaddr_in6));
	} else {
		ret = ::bind(_socket, address.get_sock_addr(), sizeof(sockaddr_in));
	}

	if (ret != 0) {
#if defined(_WIN64) || defined(_WIN32)
		return WSAGetLastError();
#else
		return errno;
#endif
	}

	return ret;
}

int Socket::listen() {
	ERR_FAIL_COND_V(_socket == 0, -1);

	return ::listen(_socket, SOMAXCONN);
}

int Socket::accept(Socket *sock) {
	ERR_FAIL_COND_V(!sock, -1);

	struct sockaddr_in6 addr6;
	memset(&addr6, 0, sizeof(addr6));
	socklen_t size = sizeof(addr6);

#ifdef __linux__
	int connfd = ::accept4(_socket, (struct sockaddr *)&addr6, &size, SOCK_NONBLOCK | SOCK_CLOEXEC);
#else
	int connfd = static_cast<int>(::accept(_socket, (struct sockaddr *)&addr6, &size));
#endif

	if (connfd >= 0) {
		sock->_socket = connfd;
		sock->_address.set_sock_addr_inet6(addr6);
#ifndef __linux__
		sock->set_non_block();
		sock->set_close_on_exit();
#endif
	}

	return connfd;
}

int Socket::close_write() {
	ERR_FAIL_COND_V(_socket == 0, -1);

#if !defined(_WIN64) && !defined(_WIN32)
	return ::shutdown(_socket, SHUT_WR);
#else
	return ::shutdown(_socket, SD_SEND);
#endif
}

int Socket::read(char *buffer, uint64_t len) {
	//ERR_FAIL_COND_V(_socket == 0, -1);

#if !defined(_WIN64) && !defined(_WIN32)
	return ::read(_socket, buffer, len);
#else
	return recv(_socket, buffer, static_cast<int>(len), 0);
#endif
}

int Socket::send(const char *buffer, uint64_t len) {
	//ERR_FAIL_COND_V(_socket == 0, -1);

#if !defined(_WIN64) && !defined(_WIN32)
	return write(_socket, buffer, len);
#else
	errno = 0;
	return ::send(_socket, buffer, static_cast<int>(len), 0);
#endif
}

void Socket::set_tcp_nodelay(bool on) {
	ERR_FAIL_COND(_socket == 0);

#if defined(_WIN64) || defined(_WIN32)
	char optval = on ? 1 : 0;
#else
	int optval = on ? 1 : 0;
#endif
	::setsockopt(_socket, IPPROTO_TCP, TCP_NODELAY, &optval, static_cast<socklen_t>(sizeof optval));
}

void Socket::set_reuse_addr(bool on) {
	ERR_FAIL_COND(_socket == 0);

#if defined(_WIN64) || defined(_WIN32)
	char optval = on ? 1 : 0;
#else
	int optval = on ? 1 : 0;
#endif
	::setsockopt(_socket, SOL_SOCKET, SO_REUSEADDR, &optval, static_cast<socklen_t>(sizeof optval));
}

int Socket::set_reuse_port(bool on) {
	ERR_FAIL_COND_V(_socket == 0, -1);

#ifdef SO_REUSEPORT
#if defined(_WIN64) || defined(_WIN32)
	char optval = on ? 1 : 0;
#else
	int optval = on ? 1 : 0;
#endif
	int ret = ::setsockopt(_socket, SOL_SOCKET, SO_REUSEPORT, &optval, static_cast<socklen_t>(sizeof optval));

	return ret;
#else
	if (on) {
		//LOG_ERR("SO_REUSEPORT is not supported.");
		return -1;
	}

	return 0;
#endif
}

void Socket::set_keep_alive(bool on) {
	ERR_FAIL_COND(_socket == 0);

#if defined(_WIN64) || defined(_WIN32)
	char optval = on ? 1 : 0;
#else
	int optval = on ? 1 : 0;
#endif
	::setsockopt(_socket, SOL_SOCKET, SO_KEEPALIVE, &optval, static_cast<socklen_t>(sizeof optval));
}

struct sockaddr_in6 Socket::get_local_addr(int *r_err) {
	struct sockaddr_in6 localaddr = { 0 };

	ERR_FAIL_COND_V(_socket == 0, localaddr);

	memset(&localaddr, 0, sizeof(localaddr));
	socklen_t addrlen = static_cast<socklen_t>(sizeof localaddr);

	int err = ::getsockname(_socket, static_cast<struct sockaddr *>((void *)(&localaddr)), &addrlen);

	if (r_err) {
		*r_err = err;
	}

	return localaddr;
}

struct sockaddr_in6 Socket::get_peer_addr(int *r_err) {
	struct sockaddr_in6 peeraddr = { 0 };

	ERR_FAIL_COND_V(_socket == 0, peeraddr);

	memset(&peeraddr, 0, sizeof(peeraddr));
	socklen_t addrlen = static_cast<socklen_t>(sizeof peeraddr);

	int err = ::getpeername(_socket, static_cast<struct sockaddr *>((void *)(&peeraddr)), &addrlen);

	if (r_err) {
		*r_err = err;
	}

	return peeraddr;
}

int Socket::global_init() {
#if defined(_WIN64) || defined(_WIN32)
	int r;
	WSADATA wsa_data;

	r = WSAStartup(MAKEWORD(2, 2), &wsa_data);

	return r;
#else
	return 0;
#endif
}

Socket::Socket() {
	_socket = 0;
}

Socket::Socket(int socketFD, const InetAddress &address) {
	_socket = socketFD;
	_address = address;
}

Socket::~Socket() {
	if (_socket >= 0) {
		close_socket();
	}
}
#line 0

#line 1 "sfw/core/sub_process.cpp"
/*************************************************************************/
/*  sub_process.cpp                                                      */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

#if defined(_WIN64) || defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

typedef struct tagLOGCONTEXTW {
	WCHAR lcName[40];
	UINT lcOptions;
	UINT lcStatus;
	UINT lcLocks;
	UINT lcMsgBase;
	UINT lcDevice;
	UINT lcPktRate;
	DWORD lcPktData;
	DWORD lcPktMode;
	DWORD lcMoveMask;
	DWORD lcBtnDnMask;
	DWORD lcBtnUpMask;
	LONG lcInOrgX;
	LONG lcInOrgY;
	LONG lcInOrgZ;
	LONG lcInExtX;
	LONG lcInExtY;
	LONG lcInExtZ;
	LONG lcOutOrgX;
	LONG lcOutOrgY;
	LONG lcOutOrgZ;
	LONG lcOutExtX;
	LONG lcOutExtY;
	LONG lcOutExtZ;
	DWORD lcSensX;
	DWORD lcSensY;
	DWORD lcSensZ;
	BOOL lcSysMode;
	int lcSysOrgX;
	int lcSysOrgY;
	int lcSysExtX;
	int lcSysExtY;
	DWORD lcSysSensX;
	DWORD lcSysSensY;
} LOGCONTEXTW;

typedef HANDLE(WINAPI *WTOpenPtr)(HWND p_window, LOGCONTEXTW *p_ctx, BOOL p_enable);

// TODO clean these up
#include <avrt.h>
#include <direct.h>
#include <knownfolders.h>
#include <process.h>
#include <regstr.h>
#include <shlobj.h>
#include <wchar.h>

struct SubProcess::SubProcessWindowsData {
	struct ProcessInfo {
		STARTUPINFO si;
		PROCESS_INFORMATION pi;
	};

	HANDLE _pipe_handles[2];
	ProcessInfo _process_info;
};

Error SubProcess::start() {
	if (_executable_path.empty()) {
		return ERR_FILE_BAD_PATH;
	}

	if (is_process_running()) {
		return ERR_BUSY;
	}

	String path = _executable_path.replace("/", "\\");

	String cmdline = _quote_command_line_argument(path);
	for (int i = 0; i < _arguments.size(); ++i) {
		cmdline += " " + _quote_command_line_argument(_arguments[i]);
	}

	ZeroMemory(&_data->_process_info.si, sizeof(_data->_process_info.si));
	_data->_process_info.si.cb = sizeof(_data->_process_info.si);
	ZeroMemory(&_data->_process_info.pi, sizeof(_data->_process_info.pi));
	LPSTARTUPINFOW si_w = (LPSTARTUPINFOW)&_data->_process_info.si;

	Char16String modstr = cmdline.utf16(); // Windows wants to change this no idea why.

	bool inherit_handles = false;

	if (_read_output) {
		// Create pipe for StdOut and StdErr.
		SECURITY_ATTRIBUTES sa;
		sa.nLength = sizeof(SECURITY_ATTRIBUTES);
		sa.bInheritHandle = true;
		sa.lpSecurityDescriptor = NULL;

		ERR_FAIL_COND_V(!CreatePipe(&_data->_pipe_handles[0], &_data->_pipe_handles[1], &sa, 0), ERR_CANT_FORK);
		ERR_FAIL_COND_V(!SetHandleInformation(_data->_pipe_handles[0], HANDLE_FLAG_INHERIT, 0), ERR_CANT_FORK); // Read handle is for host process only and should not be inherited.

		_data->_process_info.si.dwFlags |= STARTF_USESTDHANDLES;
		_data->_process_info.si.hStdOutput = _data->_pipe_handles[1];
		if (_read_std_err) {
			_data->_process_info.si.hStdError = _data->_pipe_handles[1];
		}
		inherit_handles = true;
	}

	DWORD creaton_flags = NORMAL_PRIORITY_CLASS;
	if (_open_console) {
		creaton_flags |= CREATE_NEW_CONSOLE;
	} else {
		creaton_flags |= CREATE_NO_WINDOW;
	}

	int ret = CreateProcessW(nullptr, (LPWSTR)(modstr.ptrw()), nullptr, nullptr, inherit_handles, creaton_flags, nullptr, nullptr, si_w, &_data->_process_info.pi);
	if (!ret && _read_output) {
		CloseHandle(_data->_pipe_handles[0]); // Cleanup pipe handles.
		CloseHandle(_data->_pipe_handles[1]);

		_data->_pipe_handles[0] = NULL;
		_data->_pipe_handles[1] = NULL;
	}

	ERR_FAIL_COND_V(ret == 0, ERR_CANT_FORK);

	if (_blocking) {
		if (_read_output) {
			CloseHandle(_data->_pipe_handles[1]); // Close pipe write handle (only child process is writing).

			int bytes_in_buffer = 0;

			const int CHUNK_SIZE = 4096;
			DWORD read = 0;
			for (;;) { // Read StdOut and StdErr from pipe.
				_bytes.resize(bytes_in_buffer + CHUNK_SIZE);
				const bool success = ReadFile(_data->_pipe_handles[0], _bytes.ptr() + bytes_in_buffer, CHUNK_SIZE, &read, NULL);
				if (!success || read == 0) {
					break;
				}
				// Assume that all possible encodings are ASCII-compatible.
				// Break at newline to allow receiving long output in portions.
				int newline_index = -1;
				for (int i = read - 1; i >= 0; i--) {
					if (_bytes[bytes_in_buffer + i] == '\n') {
						newline_index = i;
						break;
					}
				}

				if (newline_index == -1) {
					bytes_in_buffer += read;
					continue;
				}

				const int bytes_to_convert = bytes_in_buffer + (newline_index + 1);
				_append_to_pipe(_bytes.ptr(), bytes_to_convert);

				bytes_in_buffer = read - (newline_index + 1);
				memmove(_bytes.ptr(), _bytes.ptr() + bytes_to_convert, bytes_in_buffer);
			}

			if (bytes_in_buffer > 0) {
				_append_to_pipe(_bytes.ptr(), bytes_in_buffer);
			}

			CloseHandle(_data->_pipe_handles[0]); // Close pipe read handle.
		}

		WaitForSingleObject(_data->_process_info.pi.hProcess, INFINITE);

		DWORD ret2;
		GetExitCodeProcess(_data->_process_info.pi.hProcess, &ret2);
		_exitcode = ret2;

		CloseHandle(_data->_process_info.pi.hProcess);
		CloseHandle(_data->_process_info.pi.hThread);
	} else {
		if (_read_output) {
			//eventually we will need to keep this
			CloseHandle(_data->_pipe_handles[1]); // Close pipe write handle (only child process is writing).
			_data->_pipe_handles[1] = NULL;
		}

		_process_started = true;

		ProcessID pid = _data->_process_info.pi.dwProcessId;
		_process_id = pid;
	}

	return OK;
}

Error SubProcess::stop() {
	if (!_process_started) {
		return OK;
	}

	if (_data->_pipe_handles[0]) {
		CloseHandle(_data->_pipe_handles[0]); // Cleanup pipe handles.
		_data->_pipe_handles[0] = NULL;
	}

	if (_data->_pipe_handles[1]) {
		CloseHandle(_data->_pipe_handles[1]);
		_data->_pipe_handles[1] = NULL;
	}

	const int ret = TerminateProcess(_data->_process_info.pi.hProcess, 0);

	CloseHandle(_data->_process_info.pi.hProcess);
	CloseHandle(_data->_process_info.pi.hThread);

	ZeroMemory(&_data->_process_info.si, sizeof(_data->_process_info.si));
	_data->_process_info.si.cb = sizeof(_data->_process_info.si);
	ZeroMemory(&_data->_process_info.pi, sizeof(_data->_process_info.pi));

	_process_started = false;

	return ret != 0 ? OK : FAILED;
}

Error SubProcess::poll() {
	if (!_process_started) {
		return FAILED;
	}

	if (!_data->_pipe_handles[0]) {
		return FAILED;
	}

	_pipe.clear();

	int bytes_in_buffer = 0;

	const int CHUNK_SIZE = 4096;
	DWORD read = 0;

	_bytes.resize(bytes_in_buffer + CHUNK_SIZE);
	const bool success = ReadFile(_data->_pipe_handles[0], _bytes.ptr() + bytes_in_buffer, CHUNK_SIZE, &read, NULL);

	if (!success) {
		stop();
		return ERR_FILE_EOF;
	}

	if (read == 0) {
		return OK;
	}

	// Assume that all possible encodings are ASCII-compatible.
	// Break at newline to allow receiving long output in portions.
	int newline_index = -1;
	for (int i = read - 1; i >= 0; i--) {
		if (_bytes[bytes_in_buffer + i] == '\n') {
			newline_index = i;
			break;
		}
	}

	if (newline_index == -1) {
		bytes_in_buffer += read;
		return OK;
	}

	const int bytes_to_convert = bytes_in_buffer + (newline_index + 1);
	_append_to_pipe(_bytes.ptr(), bytes_to_convert);

	bytes_in_buffer = read - (newline_index + 1);
	memmove(_bytes.ptr(), _bytes.ptr() + bytes_to_convert, bytes_in_buffer);

	if (bytes_in_buffer > 0) {
		_append_to_pipe(_bytes.ptr(), bytes_in_buffer);
	}

	return OK;
}

Error SubProcess::send_signal(const int p_signal) {
	//Not Yet Impl
	ERR_FAIL_V(ERR_BUG);
}

Error SubProcess::send_data(const String &p_data) {
	//Not Yet Impl
	ERR_FAIL_V(ERR_BUG);
}

bool SubProcess::is_process_running() const {
	if (_process_id == 0) {
		return false;
	}

	if (!_process_started) {
		return false;
	}

	DWORD dw_exit_code = 0;
	if (!GetExitCodeProcess(_data->_process_info.pi.hProcess, &dw_exit_code)) {
		return false;
	}

	if (dw_exit_code != STILL_ACTIVE) {
		return false;
	}

	return true;
}

String SubProcess::_quote_command_line_argument(const String &p_text) const {
	for (int i = 0; i < p_text.size(); i++) {
		CharType c = p_text[i];
		if (c == ' ' || c == '&' || c == '(' || c == ')' || c == '[' || c == ']' || c == '{' || c == '}' || c == '^' || c == '=' || c == ';' || c == '!' || c == '\'' || c == '+' || c == ',' || c == '`' || c == '~') {
			return "\"" + p_text + "\"";
		}
	}
	return p_text;
}

void SubProcess::_append_to_pipe(char *p_bytes, int p_size) {
	// Try to convert from default ANSI code page to Unicode.
	LocalVector<wchar_t> wchars;
	int total_wchars = MultiByteToWideChar(CP_ACP, 0, p_bytes, p_size, nullptr, 0);
	if (total_wchars > 0) {
		wchars.resize(total_wchars);
		if (MultiByteToWideChar(CP_ACP, 0, p_bytes, p_size, wchars.ptr(), total_wchars) == 0) {
			wchars.clear();
		}
	}

	if (_pipe_mutex) {
		_pipe_mutex->lock();
	}
	if (wchars.empty()) {
		// Let's hope it's compatible with UTF-8.
		_pipe += String::utf8(p_bytes, p_size);
	} else {
		_pipe += String(wchars.ptr(), total_wchars);
	}
	if (_pipe_mutex) {
		_pipe_mutex->unlock();
	}
}

SubProcess::SubProcess() {
	_data = memnew(SubProcessWindowsData);

	_blocking = false;

	_read_output = true;

	_read_std = true;
	_read_std_err = false;

	_use_pipe_mutex = false;

	_pipe_mutex = NULL;

	_open_console = false;

	_process_id = ProcessID();
	_exitcode = 0;

	_data->_pipe_handles[0] = NULL;
	_data->_pipe_handles[1] = NULL;

	_process_started = false;

	ZeroMemory(&_data->_process_info.si, sizeof(_data->_process_info.si));
	_data->_process_info.si.cb = sizeof(_data->_process_info.si);
	ZeroMemory(&_data->_process_info.pi, sizeof(_data->_process_info.pi));
}
SubProcess::~SubProcess() {
	stop();

	memdelete(_data);
}

#else

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

Error SubProcess::start() {
#ifdef __EMSCRIPTEN__
	// Don't compile this code at all to avoid undefined references.
	// Actual virtual call goes to OS_JavaScript.
	ERR_FAIL_V(ERR_BUG);
#else
	if (_executable_path.empty()) {
		return ERR_FILE_BAD_PATH;
	}

	if (is_process_running()) {
		return ERR_BUSY;
	}

	if (_blocking && _read_output) {
		String argss;
		argss = "\"" + _executable_path + "\"";

		for (int i = 0; i < _arguments.size(); i++) {
			argss += String(" \"") + _arguments[i] + "\"";
		}

		if (_read_std_err) {
			argss += " 2>&1"; // Read stderr too
		} else {
			argss += " 2>/dev/null"; //silence stderr
		}
		FILE *f = popen(argss.utf8().get_data(), "r");

		ERR_FAIL_COND_V_MSG(!f, ERR_CANT_OPEN, "Cannot pipe stream from process running with following arguments '" + argss + "'.");

		char buf[65535];

		while (fgets(buf, 65535, f)) {
			if (_pipe_mutex) {
				_pipe_mutex->lock();
			}
			_pipe += String::utf8(buf);
			if (_pipe_mutex) {
				_pipe_mutex->unlock();
			}
		}
		int rv = pclose(f);

		_exitcode = WEXITSTATUS(rv);

		return OK;
	}

	if (!_blocking && _read_output) {
		String argss;
		argss = "\"" + _executable_path + "\"";

		for (int i = 0; i < _arguments.size(); i++) {
			argss += String(" \"") + _arguments[i] + "\"";
		}

		if (_read_std_err) {
			argss += " 2>&1"; // Read stderr too
		} else {
			argss += " 2>/dev/null"; //silence stderr
		}

		_process_fp = popen(argss.utf8().get_data(), "r");

		ERR_FAIL_COND_V_MSG(!_process_fp, ERR_CANT_OPEN, "Cannot pipe stream from process running with following arguments '" + argss + "'.");

		return OK;
	}

	// We just run it, no need to worry about output

	pid_t pid = fork();
	ERR_FAIL_COND_V(pid < 0, ERR_CANT_FORK);

	if (pid == 0) {
		// is child

		if (!_blocking) {
			// For non blocking calls, create a new session-ID so parent won't wait for it.
			// This ensures the process won't go zombie at end.
			setsid();
		}

		Vector<CharString> cs;
		cs.push_back(_executable_path.utf8());
		for (int i = 0; i < _arguments.size(); i++) {
			cs.push_back(_arguments[i].utf8());
		}

		Vector<char *> args;
		for (int i = 0; i < cs.size(); i++) {
			args.push_back((char *)cs[i].get_data());
		}
		args.push_back(0);

		execvp(_executable_path.utf8().get_data(), &args[0]);
		// still alive? something failed..
		fprintf(stderr, "**ERROR** SubProcess::execute - Could not create child process while executing: %s\n", _executable_path.utf8().get_data());
		raise(SIGKILL);
	}

	if (_blocking) {
		int status;
		waitpid(pid, &status, 0);

		_exitcode = WIFEXITED(status) ? WEXITSTATUS(status) : status;
	} else {
		_process_id = pid;
	}

	return OK;
#endif
}

Error SubProcess::stop() {
#ifdef __EMSCRIPTEN__
	// Don't compile this code at all to avoid undefined references.
	// Actual virtual call goes to OS_JavaScript.
	ERR_FAIL_V(ERR_BUG);
#else
	if (_process_fp) {
		int rv = pclose(_process_fp);
		_process_fp = NULL;
		_exitcode = WEXITSTATUS(rv);
		_process_id = 0;
		return OK;
	}

	if (_process_id) {
		int ret = ::kill(_process_id, SIGKILL);

		if (!ret) {
			//avoid zombie process
			int st;
			::waitpid(_process_id, &st, 0);
		}

		_process_id = 0;

		return ret ? ERR_INVALID_PARAMETER : OK;
	}

	return OK;
#endif
}

Error SubProcess::poll() {
#ifdef __EMSCRIPTEN__
	// Don't compile this code at all to avoid undefined references.
	// Actual virtual call goes to OS_JavaScript.
	ERR_FAIL_V(ERR_BUG);
#else

	if (_process_fp) {
		if (fgets(_process_buf, 65535, _process_fp)) {
			if (_pipe_mutex) {
				_pipe_mutex->lock();
			}
			_pipe = String::utf8(_process_buf);
			if (_pipe_mutex) {
				_pipe_mutex->unlock();
			}
		} else {
			// The process finished
			// Cleanup:
			stop();
			return ERR_FILE_EOF;
		}
	}

	return OK;
#endif
}

Error SubProcess::send_signal(const int p_signal) {
	//Not Yet Impl
	ERR_FAIL_V(ERR_BUG);
}

Error SubProcess::send_data(const String &p_data) {
	//Not Yet Impl
	ERR_FAIL_V(ERR_BUG);
}

bool SubProcess::is_process_running() const {
#ifdef __EMSCRIPTEN__
	// Don't compile this code at all to avoid undefined references.
	// Actual virtual call goes to OS_JavaScript.
	ERR_FAIL_V(false);
#else

	if (_process_fp) {
		return !feof(_process_fp);
	}

	if (_process_id == 0) {
		return false;
	}

	int status = 0;
	if (waitpid(_process_id, &status, WNOHANG) != 0) {
		return false;
	}

	return true;
#endif
}

SubProcess::SubProcess() {
	_blocking = false;

	_read_output = true;

	_read_std = true;
	_read_std_err = false;

	_use_pipe_mutex = false;

	_pipe_mutex = NULL;

	_open_console = false;

	_process_id = ProcessID();
	_exitcode = 0;

	_process_fp = NULL;
}
SubProcess::~SubProcess() {
	stop();
}

#endif

SubProcess *SubProcess::create() {
	return memnew(SubProcess());
}

String SubProcess::get_executable_path() const {
	return _executable_path;
}
void SubProcess::set_executable_path(const String &p_executable_path) {
	ERR_FAIL_COND(is_process_running());

	_executable_path = p_executable_path;
}

Vector<String> SubProcess::get_arguments() const {
	return _arguments;
}
void SubProcess::set_arguments(const Vector<String> &p_arguments) {
	ERR_FAIL_COND(is_process_running());

	_arguments = p_arguments;
}

bool SubProcess::get_blocking() const {
	return _blocking;
}
void SubProcess::set_blocking(const bool p_value) {
	ERR_FAIL_COND(is_process_running());

	_blocking = p_value;
}

bool SubProcess::get_read_output() const {
	return _read_output;
}
void SubProcess::set_read_output(const bool p_value) {
	ERR_FAIL_COND(is_process_running());

	_read_output = p_value;
}

bool SubProcess::get_read_std() const {
	return _read_std;
}
void SubProcess::set_read_std(const bool p_value) {
	ERR_FAIL_COND(is_process_running());

	_read_std = p_value;
}

bool SubProcess::get_read_std_err() const {
	return _read_std_err;
}
void SubProcess::set_read_std_err(const bool p_value) {
	ERR_FAIL_COND(is_process_running());

	_read_std_err = p_value;
}

bool SubProcess::get_use_pipe_mutex() const {
	return _use_pipe_mutex;
}
void SubProcess::set_use_pipe_mutex(const bool p_value) {
	ERR_FAIL_COND(is_process_running());

	_use_pipe_mutex = p_value;
}

bool SubProcess::get_open_console() const {
	return _open_console;
}
void SubProcess::set_open_console(const bool p_value) {
	ERR_FAIL_COND(is_process_running());

	_open_console = p_value;
}

Error SubProcess::run(const String &p_executable_path, const Vector<String> &p_arguments, bool p_output, bool p_blocking, bool p_read_std_err, bool p_use_pipe_mutex, bool p_open_console) {
	if (is_process_running()) {
		return ERR_ALREADY_IN_USE;
	}

	_executable_path = p_executable_path;
	_arguments = p_arguments;

	_blocking = p_blocking;

	_read_output = p_output;

	_read_std = true;
	_read_std_err = p_read_std_err;

	_use_pipe_mutex = p_use_pipe_mutex;

	_open_console = p_open_console;

	_setup_pipe_mutex();

	return start();
}

/*
SubProcess::SubProcess() {
	_blocking = false;

	_read_output = true;

	_read_std = true;
	_read_std_err = false;

	_use_pipe_mutex = false;

	_pipe_mutex = NULL;

	_open_console = false;

	_process_id = ProcessID();
	_exitcode = 0;
};
*/

void SubProcess::_setup_pipe_mutex() {
	if (_use_pipe_mutex) {
		if (!_pipe_mutex) {
			_pipe_mutex = memnew(Mutex);
		}
	} else {
		if (_pipe_mutex) {
			memdelete(_pipe_mutex);
			_pipe_mutex = NULL;
		}
	}
}
#line 0

#line 1 "sfw/core/sfw_core.cpp"

void SFWCore::setup() {
	if (_initialized) {
		return;
	}

	_initialized = true;

	StringName::setup();
	MemoryPool::setup();
}

void SFWCore::cleanup() {
	if (!_initialized) {
		return;
	}

	_initialized = false;

	StringName::cleanup();
	MemoryPool::cleanup();
}

bool SFWCore::_initialized = false;
#line 0

//===================  OBJECT SECTION  ===================

#line 1 "sfw/object/resource.cpp"

void Resource::emit_changed() {
	changed.emit(this);
}

Error Resource::load(const String &path) {
	return ERR_UNAVAILABLE;
}
Error Resource::save(const String &path) {
	return ERR_UNAVAILABLE;
}

Resource::Resource() :
		Reference() {
}

Resource::~Resource() {
}
#line 0

#line 1 "sfw/object/reference.cpp"

/*************************************************************************/
/*  reference.cpp                                                        */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

bool Reference::init_ref() {
	if (reference()) {
		if (!is_referenced() && refcount_init.unref()) {
			unreference(); // first referencing is already 1, so compensate for the ref above
		}

		return true;
	} else {
		return false;
	}
}

int Reference::reference_get_count() const {
	return refcount.get();
}

bool Reference::reference() {
	uint32_t rc_val = refcount.refval();
	bool success = rc_val != 0;

	return success;
}

bool Reference::unreference() {
	uint32_t rc_val = refcount.unrefval();
	bool die = rc_val == 0;

	return die;
}

Reference::Reference() :
		Object() {
	refcount.init();
	refcount_init.init();
}

Reference::~Reference() {
}

/*                                                           \
void WeakRef::set_obj(Object *p_object) {                    \
	//ref = p_object ? p_object->get_instance_id() : 0;      \
}                                                            \
                                                             \
void WeakRef::set_ref(const REF &p_ref) {                    \
	//ref = p_ref.is_valid() ? p_ref->get_instance_id() : 0; \
}                                                            \
                                                             \
WeakRef::WeakRef() :                                         \
		ref(0) {                                             \
}                                                            \
*/                                                           \
#line 0

#line 1 "sfw/object/object.cpp"

void Object::set(const StringName &p_name, const Variant &p_value, bool *r_valid) {
	if (p_name == CoreStringNames::get_singleton()->_meta) {
		//set_meta(p_name,p_value);
		metadata = p_value.duplicate();
		if (r_valid) {
			*r_valid = true;
		}
		return;
	}

	//something inside the object... :|
	bool success = _setv(p_name, p_value);
	if (success) {
		if (r_valid) {
			*r_valid = true;
		}
		return;
	}

	if (r_valid) {
		*r_valid = false;
	}
}

Variant Object::get(const StringName &p_name, bool *r_valid) const {
	Variant ret;

	if (p_name == CoreStringNames::get_singleton()->_meta) {
		ret = metadata;
		if (r_valid) {
			*r_valid = true;
		}
		return ret;

	} else {
		//something inside the object... :|
		bool success = _getv(p_name, ret);
		if (success) {
			if (r_valid) {
				*r_valid = true;
			}
			return ret;
		}

		if (r_valid) {
			*r_valid = false;
		}

		return Variant();
	}
}

bool Object::lt(const Variant &p_value_l, const Variant &p_value_r) {
	return p_value_l < p_value_r;
}

void Object::notification(int p_notification, bool p_reversed) {
	_notificationv(p_notification, p_reversed);
}

String Object::to_string() {
	return "[" + get_class() + ":" + itos(get_instance_id()) + "]";
}

bool Object::_predelete() {
	_predelete_ok = 1;
	notification(NOTIFICATION_PREDELETE, true);
	if (_predelete_ok) {
		_class_ptr = nullptr; //must restore so destructors can access class ptr correctly
	}
	return _predelete_ok;
}

void Object::_postinitialize() {
	_class_ptr = _get_class_namev();
	notification(NOTIFICATION_POSTINITIALIZE);
}

bool Object::has_meta(const String &p_name) const {
	return metadata.has(p_name);
}

void Object::set_meta(const String &p_name, const Variant &p_value) {
	if (p_value.get_type() == Variant::NIL) {
		metadata.erase(p_name);
		return;
	};

	metadata[p_name] = p_value;
}

Variant Object::get_meta(const String &p_name, const Variant &p_default) const {
	if (!metadata.has(p_name)) {
		return p_default;
	}
	return metadata[p_name];
}

void Object::remove_meta(const String &p_name) {
	metadata.erase(p_name);
}

void Object::cancel_free() {
	_predelete_ok = 0;
}

Object::Object() {
	_is_queued_for_deletion = false;
	_predelete_ok = 0;
	_instance_id = 0;
	_instance_id = ObjectDB::add_instance(this);
}

Object::~Object() {
}

ObjectRC *Object::_use_rc() {
	// The RC object is lazily created the first time it's requested;
	// that way, there's no need to allocate and release it at all if this Object
	// is not being referred by any Variant at all.

	// Although when dealing with Objects from multiple threads some locking
	// mechanism should be used, this at least makes safe the case of first
	// assignment.

	ObjectRC *rc = nullptr;
	ObjectRC *const creating = reinterpret_cast<ObjectRC *>(1);
	if (unlikely(_rc.compare_exchange_strong(rc, creating, std::memory_order_acq_rel))) {
		// Not created yet
		rc = memnew(ObjectRC(this));
		_rc.store(rc, std::memory_order_release);
		return rc;
	}

	// Spin-wait until we know it's created (or just return if it's already created)
	for (;;) {
		if (likely(rc != creating)) {
			rc->increment();
			return rc;
		}
		rc = _rc.load(std::memory_order_acquire);
	}
}

bool predelete_handler(Object *p_object) {
	return p_object->_predelete();
}

void postinitialize_handler(Object *p_object) {
	p_object->_postinitialize();
}

HashMap<ObjectID, Object *> ObjectDB::instances;
ObjectID ObjectDB::instance_counter = 1;
HashMap<Object *, ObjectID, ObjectDB::ObjectPtrHash> ObjectDB::instance_checks;
ObjectID ObjectDB::add_instance(Object *p_object) {
	ERR_FAIL_COND_V(p_object->get_instance_id() != 0, 0);

	rw_lock.write_lock();
	ObjectID instance_id = ++instance_counter;
	instances[instance_id] = p_object;
	instance_checks[p_object] = instance_id;

	rw_lock.write_unlock();

	return instance_id;
}

void ObjectDB::remove_instance(Object *p_object) {
	rw_lock.write_lock();

	instances.erase(p_object->get_instance_id());
	instance_checks.erase(p_object);

	rw_lock.write_unlock();
}
Object *ObjectDB::get_instance(ObjectID p_instance_id) {
	rw_lock.read_lock();
	Object **obj = instances.getptr(p_instance_id);
	rw_lock.read_unlock();

	if (!obj) {
		return nullptr;
	}
	return *obj;
}

void ObjectDB::debug_objects(DebugFunc p_func) {
	rw_lock.read_lock();

	const ObjectID *K = nullptr;
	while ((K = instances.next(K))) {
		p_func(instances[*K]);
	}

	rw_lock.read_unlock();
}

int ObjectDB::get_object_count() {
	rw_lock.read_lock();
	int count = instances.size();
	rw_lock.read_unlock();

	return count;
}

RWLock ObjectDB::rw_lock;

void ObjectDB::cleanup() {
	rw_lock.write_lock();
	if (instances.size()) {
		LOG_WARN("ObjectDB instances leaked at exit!");
	}
	instances.clear();
	instance_checks.clear();
	rw_lock.write_unlock();
}
#line 0

#line 1 "sfw/object/ref_ptr.cpp"
/*************************************************************************/
/*  ref_ptr.cpp                                                          */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

void RefPtr::operator=(const RefPtr &p_other) {
	Ref<Reference> *ref = reinterpret_cast<Ref<Reference> *>(&data[0]);
	Ref<Reference> *ref_other = reinterpret_cast<Ref<Reference> *>(const_cast<char *>(&p_other.data[0]));

	*ref = *ref_other;
}

bool RefPtr::operator==(const RefPtr &p_other) const {
	Ref<Reference> *ref = reinterpret_cast<Ref<Reference> *>(&data[0]);
	Ref<Reference> *ref_other = reinterpret_cast<Ref<Reference> *>(const_cast<char *>(&p_other.data[0]));

	return *ref == *ref_other;
}

bool RefPtr::operator!=(const RefPtr &p_other) const {
	Ref<Reference> *ref = reinterpret_cast<Ref<Reference> *>(&data[0]);
	Ref<Reference> *ref_other = reinterpret_cast<Ref<Reference> *>(const_cast<char *>(&p_other.data[0]));

	return *ref != *ref_other;
}

RefPtr::RefPtr(const RefPtr &p_other) {
	memnew_placement(&data[0], Ref<Reference>);

	Ref<Reference> *ref = reinterpret_cast<Ref<Reference> *>(&data[0]);
	Ref<Reference> *ref_other = reinterpret_cast<Ref<Reference> *>(const_cast<char *>(&p_other.data[0]));

	*ref = *ref_other;
}

bool RefPtr::is_null() const {
	Ref<Reference> *ref = reinterpret_cast<Ref<Reference> *>(&data[0]);
	return ref->is_null();
}

void RefPtr::unref() {
	Ref<Reference> *ref = reinterpret_cast<Ref<Reference> *>(&data[0]);
	ref->unref();
}

RefPtr::RefPtr() {
	ERR_FAIL_COND(sizeof(Ref<Reference>) > DATASIZE);
	memnew_placement(&data[0], Ref<Reference>);
}

RefPtr::~RefPtr() {
	Ref<Reference> *ref = reinterpret_cast<Ref<Reference> *>(&data[0]);
	ref->~Ref<Reference>();
}
#line 0

#line 1 "sfw/object/core_string_names.cpp"
/*************************************************************************/
/*  core_string_names.cpp                                                */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

CoreStringNames *CoreStringNames::singleton = nullptr;

CoreStringNames::CoreStringNames() :
		_free(StaticCString::create("free")),
		changed(StaticCString::create("changed")),
		_meta(StaticCString::create("__meta__")),
		_script(StaticCString::create("script")),
		script_changed(StaticCString::create("script_changed")),
		___pdcdata(StaticCString::create("___pdcdata")),
		__getvar(StaticCString::create("__getvar")),
		_iter_init(StaticCString::create("_iter_init")),
		_iter_next(StaticCString::create("_iter_next")),
		_iter_get(StaticCString::create("_iter_get")),
		get_rid(StaticCString::create("get_rid")),
		_to_string(StaticCString::create("_to_string")),
#ifdef TOOLS_ENABLED
		_sections_unfolded(StaticCString::create("_sections_unfolded")),
#endif
		_custom_features(StaticCString::create("_custom_features")),
		x(StaticCString::create("x")),
		y(StaticCString::create("y")),
		z(StaticCString::create("z")),
		w(StaticCString::create("w")),
		r(StaticCString::create("r")),
		g(StaticCString::create("g")),
		b(StaticCString::create("b")),
		a(StaticCString::create("a")),
		position(StaticCString::create("position")),
		size(StaticCString::create("size")),
		end(StaticCString::create("end")),
		basis(StaticCString::create("basis")),
		origin(StaticCString::create("origin")),
		normal(StaticCString::create("normal")),
		d(StaticCString::create("d")),
		h(StaticCString::create("h")),
		s(StaticCString::create("s")),
		v(StaticCString::create("v")),
		r8(StaticCString::create("r8")),
		g8(StaticCString::create("g8")),
		b8(StaticCString::create("b8")),
		a8(StaticCString::create("a8")) {
}
#line 0

#line 1 "sfw/object/dictionary.cpp"
/*************************************************************************/
/*  dictionary.cpp                                                       */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

struct DictionaryPrivate {
	SafeRefCount refcount;
	OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator> variant_map;
};

void Dictionary::get_key_list(List<Variant> *p_keys) const {
	if (_p->variant_map.empty()) {
		return;
	}

	for (OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::Element E = _p->variant_map.front(); E; E = E.next()) {
		p_keys->push_back(E.key());
	}
}

Variant Dictionary::get_key_at_index(int p_index) const {
	int index = 0;
	for (OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::Element E = _p->variant_map.front(); E; E = E.next()) {
		if (index == p_index) {
			return E.key();
		}
		index++;
	}

	return Variant();
}

Variant Dictionary::get_value_at_index(int p_index) const {
	int index = 0;
	for (OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::Element E = _p->variant_map.front(); E; E = E.next()) {
		if (index == p_index) {
			return E.value();
		}
		index++;
	}

	return Variant();
}

Variant &Dictionary::operator[](const Variant &p_key) {
	return _p->variant_map[p_key];
}

const Variant &Dictionary::operator[](const Variant &p_key) const {
	return _p->variant_map[p_key];
}
const Variant *Dictionary::getptr(const Variant &p_key) const {
	OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::ConstElement E = ((const OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator> *)&_p->variant_map)->find(p_key);

	if (!E) {
		return nullptr;
	}
	return &E.get();
}

Variant *Dictionary::getptr(const Variant &p_key) {
	OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::Element E = _p->variant_map.find(p_key);

	if (!E) {
		return nullptr;
	}
	return &E.get();
}

Variant Dictionary::get_valid(const Variant &p_key) const {
	OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::ConstElement E = ((const OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator> *)&_p->variant_map)->find(p_key);

	if (!E) {
		return Variant();
	}
	return E.get();
}

Variant Dictionary::get(const Variant &p_key, const Variant &p_default) const {
	const Variant *result = getptr(p_key);
	if (!result) {
		return p_default;
	}

	return *result;
}

int Dictionary::size() const {
	return _p->variant_map.size();
}
bool Dictionary::empty() const {
	return !_p->variant_map.size();
}

bool Dictionary::has(const Variant &p_key) const {
	return _p->variant_map.has(p_key);
}

bool Dictionary::has_all(const Array &p_keys) const {
	for (int i = 0; i < p_keys.size(); i++) {
		if (!has(p_keys[i])) {
			return false;
		}
	}
	return true;
}

Variant Dictionary::find_key(const Variant &p_value) const {
	for (OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::Element E = _p->variant_map.front(); E; E = E.next()) {
		if (E.value() == p_value) {
			return E.key();
		}
	}
	return Variant();
}

bool Dictionary::erase(const Variant &p_key) {
	return _p->variant_map.erase(p_key);
}

bool Dictionary::deep_equal(const Dictionary &p_dictionary, int p_recursion_count) const {
	// Cheap checks
	ERR_FAIL_COND_V_MSG(p_recursion_count > MAX_RECURSION, 0, "Max recursion reached");
	if (_p == p_dictionary._p) {
		return true;
	}
	if (_p->variant_map.size() != p_dictionary._p->variant_map.size()) {
		return false;
	}

	// Heavy O(n) check
	OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::Element this_E = _p->variant_map.front();
	OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::Element other_E = p_dictionary._p->variant_map.front();
	p_recursion_count++;
	while (this_E && other_E) {
		if (
				!this_E.key().deep_equal(other_E.key(), p_recursion_count) ||
				!this_E.value().deep_equal(other_E.value(), p_recursion_count)) {
			return false;
		}

		this_E = this_E.next();
		other_E = other_E.next();
	}

	return !this_E && !other_E;
}

bool Dictionary::operator==(const Dictionary &p_dictionary) const {
	return _p == p_dictionary._p;
}

bool Dictionary::operator!=(const Dictionary &p_dictionary) const {
	return _p != p_dictionary._p;
}

void Dictionary::_ref(const Dictionary &p_from) const {
	//make a copy first (thread safe)
	if (!p_from._p->refcount.ref()) {
		return; // couldn't copy
	}

	//if this is the same, unreference the other one
	if (p_from._p == _p) {
		_p->refcount.unref();
		return;
	}
	if (_p) {
		_unref();
	}
	_p = p_from._p;
}

void Dictionary::clear() {
	_p->variant_map.clear();
}

void Dictionary::merge(const Dictionary &p_dictionary, bool p_overwrite) {
	for (OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::Element E = p_dictionary._p->variant_map.front(); E; E = E.next()) {
		if (p_overwrite || !has(E.key())) {
			this->operator[](E.key()) = E.value();
		}
	}
}

void Dictionary::_unref() const {
	ERR_FAIL_COND(!_p);
	if (_p->refcount.unref()) {
		memdelete(_p);
	}
	_p = nullptr;
}

uint32_t Dictionary::hash() const {
	return recursive_hash(0);
}

uint32_t Dictionary::recursive_hash(int p_recursion_count) const {
	ERR_FAIL_COND_V_MSG(p_recursion_count > MAX_RECURSION, 0, "Max recursion reached");
	p_recursion_count++;

	uint32_t h = hash_murmur3_one_32(Variant::DICTIONARY);

	for (OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::Element E = _p->variant_map.front(); E; E = E.next()) {
		h = hash_murmur3_one_32(E.key().recursive_hash(p_recursion_count), h);
		h = hash_murmur3_one_32(E.value().recursive_hash(p_recursion_count), h);
	}

	return hash_fmix32(h);
}

Array Dictionary::keys() const {
	Array varr;
	if (_p->variant_map.empty()) {
		return varr;
	}

	varr.resize(size());

	int i = 0;
	for (OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::Element E = _p->variant_map.front(); E; E = E.next()) {
		varr[i] = E.key();
		i++;
	}

	return varr;
}

Array Dictionary::values() const {
	Array varr;
	if (_p->variant_map.empty()) {
		return varr;
	}

	varr.resize(size());

	int i = 0;
	for (OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::Element E = _p->variant_map.front(); E; E = E.next()) {
		varr[i] = E.get();
		i++;
	}

	return varr;
}

const Variant *Dictionary::next(const Variant *p_key) const {
	if (p_key == nullptr) {
		// caller wants to get the first element
		if (_p->variant_map.front()) {
			return &_p->variant_map.front().key();
		}
		return nullptr;
	}
	OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::Element E = _p->variant_map.find(*p_key);

	if (E && E.next()) {
		return &E.next().key();
	}
	return nullptr;
}

Dictionary Dictionary::duplicate(bool p_deep) const {
	Dictionary n;

	for (OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::Element E = _p->variant_map.front(); E; E = E.next()) {
		n[E.key()] = p_deep ? E.value().duplicate(true) : E.value();
	}

	return n;
}

void Dictionary::operator=(const Dictionary &p_dictionary) {
	_ref(p_dictionary);
}

const void *Dictionary::id() const {
	return _p;
}

Dictionary::Dictionary(const Dictionary &p_from) {
	_p = nullptr;
	_ref(p_from);
}

Dictionary::Dictionary() {
	_p = memnew(DictionaryPrivate);
	_p->refcount.init();
}
Dictionary::~Dictionary() {
	_unref();
}
#line 0

#line 1 "sfw/object/array.cpp"
/*************************************************************************/
/*  array.cpp                                                            */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

class ArrayPrivate {
public:
	SafeRefCount refcount;
	Vector<Variant> array;
};

void Array::_ref(const Array &p_from) const {
	ArrayPrivate *_fp = p_from._p;

	ERR_FAIL_COND(!_fp); // should NOT happen.

	if (_fp == _p) {
		return; // whatever it is, nothing to do here move along
	}

	bool success = _fp->refcount.ref();

	ERR_FAIL_COND(!success); // should really not happen either

	_unref();

	_p = p_from._p;
}

void Array::_unref() const {
	if (!_p) {
		return;
	}

	if (_p->refcount.unref()) {
		memdelete(_p);
	}
	_p = nullptr;
}

Variant &Array::operator[](int p_idx) {
	return _p->array.write[p_idx];
}

const Variant &Array::operator[](int p_idx) const {
	return _p->array[p_idx];
}

int Array::size() const {
	return _p->array.size();
}
bool Array::empty() const {
	return _p->array.empty();
}
void Array::clear() {
	_p->array.clear();
}

bool Array::deep_equal(const Array &p_array, int p_recursion_count) const {
	// Cheap checks
	ERR_FAIL_COND_V_MSG(p_recursion_count > MAX_RECURSION, true, "Max recursion reached");
	if (_p == p_array._p) {
		return true;
	}
	const Vector<Variant> &a1 = _p->array;
	const Vector<Variant> &a2 = p_array._p->array;
	const int size = a1.size();
	if (size != a2.size()) {
		return false;
	}

	// Heavy O(n) check
	p_recursion_count++;
	for (int i = 0; i < size; i++) {
		if (!a1[i].deep_equal(a2[i], p_recursion_count)) {
			return false;
		}
	}

	return true;
}

bool Array::operator==(const Array &p_array) const {
	return _p == p_array._p;
}

uint32_t Array::hash() const {
	return recursive_hash(0);
}

uint32_t Array::recursive_hash(int p_recursion_count) const {
	ERR_FAIL_COND_V_MSG(p_recursion_count > MAX_RECURSION, 0, "Max recursion reached");
	p_recursion_count++;

	uint32_t h = hash_murmur3_one_32(0);

	for (int i = 0; i < _p->array.size(); i++) {
		h = hash_murmur3_one_32(_p->array[i].recursive_hash(p_recursion_count), h);
	}
	return hash_fmix32(h);
}

void Array::operator=(const Array &p_array) {
	_ref(p_array);
}

void Array::push_back(const Variant &p_value) {
	_p->array.push_back(p_value);
}

void Array::append_array(const Array &p_array) {
	_p->array.append_array(p_array._p->array);
}

Error Array::resize(int p_new_size) {
	return _p->array.resize(p_new_size);
}

void Array::insert(int p_pos, const Variant &p_value) {
	_p->array.insert(p_pos, p_value);
}

void Array::fill(const Variant &p_value) {
	_p->array.fill(p_value);
}

void Array::erase(const Variant &p_value) {
	_p->array.erase(p_value);
}

Variant Array::front() const {
	ERR_FAIL_COND_V_MSG(_p->array.size() == 0, Variant(), "Can't take value from empty array.");
	return operator[](0);
}

Variant Array::back() const {
	ERR_FAIL_COND_V_MSG(_p->array.size() == 0, Variant(), "Can't take value from empty array.");
	return operator[](_p->array.size() - 1);
}

int Array::find(const Variant &p_value, int p_from) const {
	return _p->array.find(p_value, p_from);
}

int Array::rfind(const Variant &p_value, int p_from) const {
	if (_p->array.size() == 0) {
		return -1;
	}

	if (p_from < 0) {
		// Relative offset from the end
		p_from = _p->array.size() + p_from;
	}
	if (p_from < 0 || p_from >= _p->array.size()) {
		// Limit to array boundaries
		p_from = _p->array.size() - 1;
	}

	for (int i = p_from; i >= 0; i--) {
		if (_p->array[i] == p_value) {
			return i;
		}
	}

	return -1;
}

int Array::find_last(const Variant &p_value) const {
	return rfind(p_value);
}

int Array::count(const Variant &p_value) const {
	if (_p->array.size() == 0) {
		return 0;
	}

	int amount = 0;
	for (int i = 0; i < _p->array.size(); i++) {
		if (_p->array[i] == p_value) {
			amount++;
		}
	}

	return amount;
}

bool Array::has(const Variant &p_value) const {
	return _p->array.find(p_value, 0) != -1;
}

void Array::remove(int p_pos) {
	_p->array.remove(p_pos);
}

void Array::set(int p_idx, const Variant &p_value) {
	operator[](p_idx) = p_value;
}

const Variant &Array::get(int p_idx) const {
	return operator[](p_idx);
}

Array Array::duplicate(bool p_deep) const {
	Array new_arr;
	int element_count = size();
	new_arr.resize(element_count);
	for (int i = 0; i < element_count; i++) {
		new_arr[i] = p_deep ? get(i).duplicate(p_deep) : get(i);
	}

	return new_arr;
}

int Array::_clamp_slice_index(int p_index) const {
	int arr_size = size();
	int fixed_index = CLAMP(p_index, -arr_size, arr_size - 1);
	if (fixed_index < 0) {
		fixed_index = arr_size + fixed_index;
	}
	return fixed_index;
}

Array Array::slice(int p_begin, int p_end, int p_step, bool p_deep) const { // like python, but inclusive on upper bound

	Array new_arr;

	ERR_FAIL_COND_V_MSG(p_step == 0, new_arr, "Array slice step size cannot be zero.");

	if (empty()) { // Don't try to slice empty arrays.
		return new_arr;
	}
	if (p_step > 0) {
		if (p_begin >= size() || p_end < -size()) {
			return new_arr;
		}
	} else { // p_step < 0
		if (p_begin < -size() || p_end >= size()) {
			return new_arr;
		}
	}

	int begin = _clamp_slice_index(p_begin);
	int end = _clamp_slice_index(p_end);

	int new_arr_size = MAX(((end - begin + p_step) / p_step), 0);
	new_arr.resize(new_arr_size);

	if (p_step > 0) {
		int dest_idx = 0;
		for (int idx = begin; idx <= end; idx += p_step) {
			ERR_FAIL_COND_V_MSG(dest_idx < 0 || dest_idx >= new_arr_size, Array(), "Bug in Array slice()");
			new_arr[dest_idx++] = p_deep ? get(idx).duplicate(p_deep) : get(idx);
		}
	} else { // p_step < 0
		int dest_idx = 0;
		for (int idx = begin; idx >= end; idx += p_step) {
			ERR_FAIL_COND_V_MSG(dest_idx < 0 || dest_idx >= new_arr_size, Array(), "Bug in Array slice()");
			new_arr[dest_idx++] = p_deep ? get(idx).duplicate(p_deep) : get(idx);
		}
	}

	return new_arr;
}

struct _ArrayVariantSort {
	_FORCE_INLINE_ bool operator()(const Variant &p_l, const Variant &p_r) const {
		bool valid = false;
		Variant res;
		Variant::evaluate(Variant::OP_LESS, p_l, p_r, res, valid);
		if (!valid) {
			res = false;
		}
		return res;
	}
};

Array &Array::sort() {
	_p->array.sort_custom<_ArrayVariantSort>();
	return *this;
}

struct _ArrayVariantSortCustom {
	Object *obj;

	_FORCE_INLINE_ bool operator()(const Variant &p_l, const Variant &p_r) const {
		return obj->lt(p_l, p_r);
	}
};
Array &Array::sort_custom(Object *p_obj) {
	ERR_FAIL_COND_V(!p_obj, *this);

	SortArray<Variant, _ArrayVariantSortCustom, true> avs;
	avs.compare.obj = p_obj;
	avs.sort(_p->array.ptrw(), _p->array.size());
	return *this;
}

void Array::shuffle() {
	const int n = _p->array.size();
	if (n < 2) {
		return;
	}
	Variant *data = _p->array.ptrw();
	for (int i = n - 1; i >= 1; i--) {
		const int j = Math::rand() % (i + 1);
		const Variant tmp = data[j];
		data[j] = data[i];
		data[i] = tmp;
	}
}

template <typename Less>
_FORCE_INLINE_ int bisect(const Vector<Variant> &p_array, const Variant &p_value, bool p_before, const Less &p_less) {
	int lo = 0;
	int hi = p_array.size();
	if (p_before) {
		while (lo < hi) {
			const int mid = (lo + hi) / 2;
			if (p_less(p_array.get(mid), p_value)) {
				lo = mid + 1;
			} else {
				hi = mid;
			}
		}
	} else {
		while (lo < hi) {
			const int mid = (lo + hi) / 2;
			if (p_less(p_value, p_array.get(mid))) {
				hi = mid;
			} else {
				lo = mid + 1;
			}
		}
	}
	return lo;
}

int Array::bsearch(const Variant &p_value, bool p_before) {
	return bisect(_p->array, p_value, p_before, _ArrayVariantSort());
}

int Array::bsearch_custom(const Variant &p_value, Object *p_obj, const StringName &p_function, bool p_before) {
	ERR_FAIL_COND_V(!p_obj, 0);

	_ArrayVariantSortCustom less;
	less.obj = p_obj;

	return bisect(_p->array, p_value, p_before, less);
}

Array &Array::invert() {
	_p->array.invert();
	return *this;
}

void Array::push_front(const Variant &p_value) {
	_p->array.insert(0, p_value);
}

Variant Array::pop_back() {
	if (!_p->array.empty()) {
		const int n = _p->array.size() - 1;
		const Variant ret = _p->array.get(n);
		_p->array.resize(n);
		return ret;
	}
	return Variant();
}

Variant Array::pop_front() {
	if (!_p->array.empty()) {
		const Variant ret = _p->array.get(0);
		_p->array.remove(0);
		return ret;
	}
	return Variant();
}

Variant Array::pop_at(int p_pos) {
	if (_p->array.empty()) {
		// Return `null` without printing an error to mimic `pop_back()` and `pop_front()` behavior.
		return Variant();
	}

	if (p_pos < 0) {
		// Relative offset from the end
		p_pos = _p->array.size() + p_pos;
	}

	ERR_FAIL_INDEX_V_MSG(
			p_pos,
			_p->array.size(),
			Variant(),
			vformat(
					"The calculated index %s is out of bounds (the array has %s elements). Leaving the array untouched and returning `null`.",
					p_pos,
					_p->array.size()));

	const Variant ret = _p->array.get(p_pos);
	_p->array.remove(p_pos);
	return ret;
}

#if defined(_WIN64) || defined(_WIN32)
// Windows...
#undef min
#undef max
#endif

Variant Array::min() const {
	Variant minval;
	for (int i = 0; i < size(); i++) {
		if (i == 0) {
			minval = get(i);
		} else {
			bool valid;
			Variant ret;
			Variant test = get(i);
			Variant::evaluate(Variant::OP_LESS, test, minval, ret, valid);
			if (!valid) {
				return Variant(); //not a valid comparison
			}
			if (bool(ret)) {
				//is less
				minval = test;
			}
		}
	}
	return minval;
}

Variant Array::max() const {
	Variant maxval;
	for (int i = 0; i < size(); i++) {
		if (i == 0) {
			maxval = get(i);
		} else {
			bool valid;
			Variant ret;
			Variant test = get(i);
			Variant::evaluate(Variant::OP_GREATER, test, maxval, ret, valid);
			if (!valid) {
				return Variant(); //not a valid comparison
			}
			if (bool(ret)) {
				//is less
				maxval = test;
			}
		}
	}
	return maxval;
}

const void *Array::id() const {
	return _p;
}

String Array::sprintf(const String &p_format, bool *error) const {
	String formatted;
	CharType *self = (CharType *)p_format.get_data();
	bool in_format = false;
	int value_index = 0;
	int min_chars = 0;
	int min_decimals = 0;
	bool in_decimals = false;
	bool pad_with_zeros = false;
	bool left_justified = false;
	bool show_sign = false;

	const Array &values = *this;

	if (error) {
		*error = true;
	}

	for (; *self; self++) {
		const CharType c = *self;

		if (in_format) { // We have % - let's see what else we get.
			switch (c) {
				case '%': { // Replace %% with %
					formatted += p_format.chr(c);
					in_format = false;
					break;
				}
				case 'd': // Integer (signed)
				case 'o': // Octal
				case 'x': // Hexadecimal (lowercase)
				case 'X': { // Hexadecimal (uppercase)
					if (value_index >= values.size()) {
						return "not enough arguments for format string";
					}

					if (!values[value_index].is_num()) {
						return "a number is required";
					}

					int64_t value = values[value_index];
					int base = 16;
					bool capitalize = false;
					switch (c) {
						case 'd':
							base = 10;
							break;
						case 'o':
							base = 8;
							break;
						case 'x':
							break;
						case 'X':
							base = 16;
							capitalize = true;
							break;
					}
					// Get basic number.
					String str = String::num_int64(ABS(value), base, capitalize);
					int number_len = str.length();

					// Padding.
					int pad_chars_count = (value < 0 || show_sign) ? min_chars - 1 : min_chars;
					String pad_char = pad_with_zeros ? String("0") : String(" ");
					if (left_justified) {
						str = str.rpad(pad_chars_count, pad_char);
					} else {
						str = str.lpad(pad_chars_count, pad_char);
					}

					// Sign.
					if (show_sign || value < 0) {
						String sign_char = value < 0 ? "-" : "+";
						if (left_justified) {
							str = str.insert(0, sign_char);
						} else {
							str = str.insert(pad_with_zeros ? 0 : str.length() - number_len, sign_char);
						}
					}

					formatted += str;
					++value_index;
					in_format = false;

					break;
				}
				case 'f': { // Float
					if (value_index >= values.size()) {
						return "not enough arguments for format string";
					}

					if (!values[value_index].is_num()) {
						return "a number is required";
					}

					double value = values[value_index];
					bool is_negative = (value < 0);
					String str = String::num(ABS(value), min_decimals);
					bool not_numeric = isinf(value) || isnan(value);

					// Pad decimals out.
					if (!not_numeric) {
						str = str.pad_decimals(min_decimals);
					}

					int initial_len = str.length();

					// Padding. Leave room for sign later if required.
					int pad_chars_count = (is_negative || show_sign) ? min_chars - 1 : min_chars;
					String pad_char = (pad_with_zeros && !not_numeric) ? String("0") : String(" "); // Never pad NaN or inf with zeros
					if (left_justified) {
						str = str.rpad(pad_chars_count, pad_char);
					} else {
						str = str.lpad(pad_chars_count, pad_char);
					}

					// Add sign if needed.
					if (show_sign || is_negative) {
						String sign_char = is_negative ? "-" : "+";
						if (left_justified) {
							str = str.insert(0, sign_char);
						} else {
							str = str.insert(pad_with_zeros ? 0 : str.length() - initial_len, sign_char);
						}
					}

					formatted += str;
					++value_index;
					in_format = false;
					break;
				}
				case 's': { // String
					if (value_index >= values.size()) {
						return "not enough arguments for format string";
					}

					String str = values[value_index];
					// Padding.
					if (left_justified) {
						str = str.rpad(min_chars);
					} else {
						str = str.lpad(min_chars);
					}

					formatted += str;
					++value_index;
					in_format = false;
					break;
				}
				case 'c': {
					if (value_index >= values.size()) {
						return "not enough arguments for format string";
					}

					// Convert to character.
					String str;
					if (values[value_index].is_num()) {
						int value = values[value_index];
						if (value < 0) {
							return "unsigned integer is lower than minimum";
						} else if (value >= 0xd800 && value <= 0xdfff) {
							return "unsigned integer is invalid Unicode character";
						} else if (value > 0x10ffff) {
							return "unsigned integer is greater than maximum";
						}
						str = p_format.chr(values[value_index]);
					} else if (values[value_index].get_type() == Variant::STRING) {
						str = values[value_index];
						if (str.length() != 1) {
							return "%c requires number or single-character string";
						}
					} else {
						return "%c requires number or single-character string";
					}

					// Padding.
					if (left_justified) {
						str = str.rpad(min_chars);
					} else {
						str = str.lpad(min_chars);
					}

					formatted += str;
					++value_index;
					in_format = false;
					break;
				}
				case '-': { // Left justify
					left_justified = true;
					break;
				}
				case '+': { // Show + if positive.
					show_sign = true;
					break;
				}
				case '0':
				case '1':
				case '2':
				case '3':
				case '4':
				case '5':
				case '6':
				case '7':
				case '8':
				case '9': {
					int n = c - '0';
					if (in_decimals) {
						min_decimals *= 10;
						min_decimals += n;
					} else {
						if (c == '0' && min_chars == 0) {
							if (left_justified) {
								LOG_WARN("'0' flag ignored with '-' flag in string format");
							} else {
								pad_with_zeros = true;
							}
						} else {
							min_chars *= 10;
							min_chars += n;
						}
					}
					break;
				}
				case '.': { // Float separator.
					if (in_decimals) {
						return "too many decimal points in format";
					}
					in_decimals = true;
					min_decimals = 0; // We want to add the value manually.
					break;
				}

				case '*': { // Dynamic width, based on value.
					if (value_index >= values.size()) {
						return "not enough arguments for format string";
					}

					if (!values[value_index].is_num()) {
						return "* wants number";
					}

					int size = values[value_index];

					if (in_decimals) {
						min_decimals = size;
					} else {
						min_chars = size;
					}

					++value_index;
					break;
				}

				default: {
					return "unsupported format character";
				}
			}
		} else { // Not in format string.
			switch (c) {
				case '%':
					in_format = true;
					// Back to defaults:
					min_chars = 0;
					min_decimals = 6;
					pad_with_zeros = false;
					left_justified = false;
					show_sign = false;
					in_decimals = false;
					break;
				default:
					formatted += p_format.chr(c);
			}
		}
	}

	if (in_format) {
		return "incomplete format";
	}

	if (value_index != values.size()) {
		return "not all arguments converted during string formatting";
	}

	if (error) {
		*error = false;
	}
	return formatted;
}

Array::Array(const Array &p_from) {
	_p = nullptr;
	_ref(p_from);
}

Array::Array() {
	_p = memnew(ArrayPrivate);
	_p->refcount.init();
}
Array::~Array() {
	_unref();
}
#line 0

#line 1 "sfw/object/psignal.cpp"

void Signal::connect_static(void (*func)(Signal *)) {
	StaticSignalEntry *se = memnew(StaticSignalEntry());
	se->func = func;

	entries.push_back(se);
}
void Signal::disconnect_static(void (*func)(Signal *)) {
	for (int i = 0; i < entries.size(); ++i) {
		SignalEntry *e = entries[i];

		if (e->type == SIGNAL_ENTRY_TYPE_STATIC) {
			StaticSignalEntry *se = static_cast<StaticSignalEntry *>(e);

			if (se->func == func) {
				entries.remove(i);
				return;
			}
		}
	}
}
bool Signal::is_connected_static(void (*func)(Signal *)) {
	for (int i = 0; i < entries.size(); ++i) {
		SignalEntry *e = entries[i];

		if (e->type == SIGNAL_ENTRY_TYPE_STATIC) {
			StaticSignalEntry *se = static_cast<StaticSignalEntry *>(e);

			if (se->func == func) {
				return true;
			}
		}
	}

	return false;
}

void Signal::emit(Object *p_emitter) {
	emitter = p_emitter;

	for (int i = 0; i < entries.size(); ++i) {
		entries[i]->call(this);
	}
}

void Signal::emit(Object *p_emitter, const Variant &p1) {
	emitter = p_emitter;

	params.push_back(p1);

	for (int i = 0; i < entries.size(); ++i) {
		entries[i]->call(this);
	}

	params.clear();
}
void Signal::emit(Object *p_emitter, const Variant &p1, const Variant &p2) {
	emitter = p_emitter;

	params.push_back(p1);
	params.push_back(p2);

	for (int i = 0; i < entries.size(); ++i) {
		entries[i]->call(this);
	}

	params.clear();
}
void Signal::emit(Object *p_emitter, const Variant &p1, const Variant &p2, const Variant &p3) {
	emitter = p_emitter;

	params.push_back(p1);
	params.push_back(p2);
	params.push_back(p3);

	for (int i = 0; i < entries.size(); ++i) {
		entries[i]->call(this);
	}

	params.clear();
}

void Signal::emit(Object *p_emitter, const Variant &p1, const Variant &p2, const Variant &p3, const Variant &p4) {
	emitter = p_emitter;

	params.push_back(p1);
	params.push_back(p2);
	params.push_back(p3);
	params.push_back(p4);

	for (int i = 0; i < entries.size(); ++i) {
		entries[i]->call(this);
	}

	params.clear();
}

void Signal::emit(Object *p_emitter, const Variant &p1, const Variant &p2, const Variant &p3, const Variant &p4, const Variant &p5) {
	emitter = p_emitter;

	params.push_back(p1);
	params.push_back(p2);
	params.push_back(p3);
	params.push_back(p4);
	params.push_back(p5);

	for (int i = 0; i < entries.size(); ++i) {
		entries[i]->call(this);
	}

	params.clear();
}

Signal::Signal() {
}
Signal::~Signal() {
}
#line 0

#line 1 "sfw/object/variant.cpp"
/*************************************************************************/
/*  variant.cpp                                                          */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

String Variant::get_type_name(Variant::Type p_type) {
	switch (p_type) {
		case NIL: {
			return "Nil";
		} break;

		// atomic types
		case BOOL: {
			return "bool";
		} break;
		case INT: {
			return "int";

		} break;
		case REAL: {
			return "float";

		} break;
		case STRING: {
			return "String";
		} break;

		// math types
		case RECT2: {
			return "Rect2";
		} break;
		case RECT2I: {
			return "Rect2i";
		} break;
		case VECTOR2: {
			return "Vector2";
		} break;
		case VECTOR2I: {
			return "Vector2i";
		} break;
		case VECTOR3: {
			return "Vector3";
		} break;
		case VECTOR3I: {
			return "Vector3i";
		} break;
		case VECTOR4: {
			return "Vector4";
		} break;
		case VECTOR4I: {
			return "Vector4i";
		} break;

		case PLANE: {
			return "Plane";
		} break;
		case QUATERNION: {
			return "Quaternion";
		} break;
		case AABB: {
			return "AABB";
		} break;
		case BASIS: {
			return "Basis";
		} break;
		case TRANSFORM: {
			return "Transform";
		} break;
		case TRANSFORM2D: {
			return "Transform2D";
		} break;
		case PROJECTION: {
			return "Projection";
		} break;

		// misc types
		case COLOR: {
			return "Color";
		} break;
		case OBJECT: {
			return "Object";
		} break;
		case STRING_NAME: {
			return "StringName";
		} break;
		case DICTIONARY: {
			return "Dictionary";
		} break;
		case ARRAY: {
			return "Array";
		} break;

		// arrays
		case POOL_BYTE_ARRAY: {
			return "PoolByteArray";
		} break;
		case POOL_INT_ARRAY: {
			return "PoolIntArray";
		} break;
		case POOL_REAL_ARRAY: {
			return "PoolRealArray";
		} break;
		case POOL_STRING_ARRAY: {
			return "PoolStringArray";
		} break;
		case POOL_VECTOR2_ARRAY: {
			return "PoolVector2Array";
		} break;
		case POOL_VECTOR2I_ARRAY: {
			return "PoolVector2iArray";
		} break;
		case POOL_VECTOR3_ARRAY: {
			return "PoolVector3Array";
		} break;
		case POOL_VECTOR3I_ARRAY: {
			return "PoolVector3iArray";
		} break;
		case POOL_VECTOR4_ARRAY: {
			return "PoolVector4Array";
		} break;
		case POOL_VECTOR4I_ARRAY: {
			return "PoolVector4iArray";
		} break;
		case POOL_COLOR_ARRAY: {
			return "PoolColorArray";
		} break;
		default: {
		}
	}

	return "";
}

bool Variant::can_convert(Variant::Type p_type_from, Variant::Type p_type_to) {
	if (p_type_from == p_type_to) {
		return true;
	}
	if (p_type_to == NIL && p_type_from != NIL) { //nil can convert to anything
		return true;
	}

	if (p_type_from == NIL) {
		return (p_type_to == OBJECT);
	};

	const Type *valid_types = nullptr;
	const Type *invalid_types = nullptr;

	switch (p_type_to) {
		case NIL: {
			//can't
		} break;
		case BOOL: {
			static const Type valid[] = {
				INT,
				REAL,
				STRING,
				NIL,
			};

			valid_types = valid;
		} break;
		case INT: {
			static const Type valid[] = {
				BOOL,
				REAL,
				STRING,
				NIL,
			};

			valid_types = valid;
		} break;
		case REAL: {
			static const Type valid[] = {
				BOOL,
				INT,
				STRING,
				NIL,
			};

			valid_types = valid;
		} break;
		case STRING: {
			static const Type invalid[] = {
				OBJECT,
				NIL
			};

			invalid_types = invalid;
		} break;
		case RECT2: {
			static const Type valid[] = {
				RECT2I,
				NIL
			};

			valid_types = valid;
		} break;
		case RECT2I: {
			static const Type valid[] = {
				RECT2,
				NIL
			};

			valid_types = valid;
		} break;
		case VECTOR2: {
			static const Type valid[] = {
				VECTOR2I,
				NIL
			};

			valid_types = valid;
		} break;
		case VECTOR2I: {
			static const Type valid[] = {
				VECTOR2,
				NIL
			};

			valid_types = valid;
		} break;
		case VECTOR3: {
			static const Type valid[] = {
				VECTOR3I,
				NIL
			};

			valid_types = valid;
		} break;
		case VECTOR3I: {
			static const Type valid[] = {
				VECTOR3,
				NIL
			};

			valid_types = valid;
		} break;
		case VECTOR4: {
			static const Type valid[] = {
				VECTOR4I,
				NIL
			};

			valid_types = valid;
		} break;
		case VECTOR4I: {
			static const Type valid[] = {
				VECTOR4,
				NIL
			};

			valid_types = valid;
		} break;
		case PLANE: {
			//can't
		} break;
		case QUATERNION: {
			static const Type valid[] = {
				BASIS,
				NIL
			};

			valid_types = valid;
		} break;
		case AABB: {
			//can't
		} break;
		case BASIS: {
			static const Type valid[] = {
				QUATERNION,
				VECTOR3,
				NIL
			};

			valid_types = valid;
		} break;
		case TRANSFORM: {
			static const Type valid[] = {
				TRANSFORM2D,
				QUATERNION,
				BASIS,
				NIL
			};

			valid_types = valid;
		} break;
		case TRANSFORM2D: {
			static const Type valid[] = {
				TRANSFORM,
				NIL
			};

			valid_types = valid;
		} break;
		case PROJECTION: {
			static const Type valid[] = {
				TRANSFORM,
				NIL
			};

			valid_types = valid;
		} break;
		// misc types
		case COLOR: {
			static const Type valid[] = {
				STRING,
				INT,
				NIL,
			};

			valid_types = valid;
		} break;
		case OBJECT: {
			static const Type valid[] = {
				NIL
			};

			valid_types = valid;
		} break;
		case STRING_NAME: {
			static const Type valid[] = {
				STRING,
				NIL
			};

			valid_types = valid;
		} break;
		case DICTIONARY: {
			//can't
		} break;
		case ARRAY: {
			static const Type valid[] = {
				POOL_BYTE_ARRAY,
				POOL_INT_ARRAY,
				POOL_STRING_ARRAY,
				POOL_REAL_ARRAY,
				POOL_COLOR_ARRAY,
				POOL_VECTOR2_ARRAY,
				POOL_VECTOR2I_ARRAY,
				POOL_VECTOR3_ARRAY,
				POOL_VECTOR3I_ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		// arrays
		case POOL_BYTE_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		case POOL_INT_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;
		} break;
		case POOL_REAL_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		case POOL_STRING_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;
		} break;
		case POOL_VECTOR2_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;
		} break;
		case POOL_VECTOR2I_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;
		} break;
		case POOL_VECTOR3_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;
		} break;
		case POOL_VECTOR3I_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;
		} break;
		case POOL_VECTOR4_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;
		} break;
		case POOL_VECTOR4I_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;
		} break;
		case POOL_COLOR_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		default: {
		}
	}

	if (valid_types) {
		int i = 0;
		while (valid_types[i] != NIL) {
			if (p_type_from == valid_types[i]) {
				return true;
			}
			i++;
		}

	} else if (invalid_types) {
		int i = 0;
		while (invalid_types[i] != NIL) {
			if (p_type_from == invalid_types[i]) {
				return false;
			}
			i++;
		}

		return true;
	}

	return false;
}

bool Variant::can_convert_strict(Variant::Type p_type_from, Variant::Type p_type_to) {
	if (p_type_from == p_type_to) {
		return true;
	}
	if (p_type_to == NIL && p_type_from != NIL) { //nil can convert to anything
		return true;
	}

	if (p_type_from == NIL) {
		return (p_type_to == OBJECT);
	};

	const Type *valid_types = nullptr;

	switch (p_type_to) {
		case NIL: {
			//can't, also already handled
		} break;
		case BOOL: {
			static const Type valid[] = {
				INT,
				REAL,
				//STRING,
				NIL,
			};

			valid_types = valid;
		} break;
		case INT: {
			static const Type valid[] = {
				BOOL,
				REAL,
				//STRING,
				NIL,
			};

			valid_types = valid;

		} break;
		case REAL: {
			static const Type valid[] = {
				BOOL,
				INT,
				//STRING,
				NIL,
			};

			valid_types = valid;

		} break;
		case STRING: {
			static const Type valid[] = {
				STRING_NAME,
				NIL
			};

			valid_types = valid;
		} break;
		case RECT2: {
			static const Type valid[] = {
				RECT2I,
				NIL
			};

			valid_types = valid;
		} break;
		case RECT2I: {
			static const Type valid[] = {
				RECT2,
				NIL
			};

			valid_types = valid;
		} break;
		case VECTOR2: {
			static const Type valid[] = {
				VECTOR2I,
				NIL
			};

			valid_types = valid;
		} break;
		case VECTOR2I: {
			static const Type valid[] = {
				VECTOR2,
				NIL
			};

			valid_types = valid;
		} break;
		case VECTOR3: {
			static const Type valid[] = {
				VECTOR3I,
				NIL
			};

			valid_types = valid;
		} break;
		case VECTOR3I: {
			static const Type valid[] = {
				VECTOR3,
				NIL
			};

			valid_types = valid;
		} break;
		case VECTOR4: {
			static const Type valid[] = {
				VECTOR4I,
				NIL
			};

			valid_types = valid;
		} break;
		case VECTOR4I: {
			static const Type valid[] = {
				VECTOR4,
				NIL
			};

			valid_types = valid;
		} break;
		case PLANE: {
			//Can't
		} break;
		case QUATERNION: {
			static const Type valid[] = {
				BASIS,
				NIL
			};

			valid_types = valid;
		} break;
		case AABB: {
			//Can't
		} break;
		case BASIS: {
			static const Type valid[] = {
				QUATERNION,
				VECTOR3,
				NIL
			};

			valid_types = valid;
		} break;
		case TRANSFORM: {
			static const Type valid[] = {
				TRANSFORM2D,
				QUATERNION,
				BASIS,
				NIL
			};

			valid_types = valid;
		} break;
		case TRANSFORM2D: {
			static const Type valid[] = {
				TRANSFORM,
				NIL
			};

			valid_types = valid;
		} break;
		case PROJECTION: {
			static const Type valid[] = {
				TRANSFORM,
				NIL
			};

			valid_types = valid;
		} break;
		case COLOR: {
			static const Type valid[] = {
				STRING,
				INT,
				NIL,
			};

			valid_types = valid;
		} break;
		case OBJECT: {
			static const Type valid[] = {
				NIL
			};

			valid_types = valid;
		} break;
		case STRING_NAME: {
			static const Type valid[] = {
				STRING,
				NIL
			};

			valid_types = valid;
		} break;
		case DICTIONARY: {
			//Can't
		} break;
		case ARRAY: {
			static const Type valid[] = {
				POOL_BYTE_ARRAY,
				POOL_INT_ARRAY,
				POOL_STRING_ARRAY,
				POOL_REAL_ARRAY,
				POOL_COLOR_ARRAY,
				POOL_VECTOR2_ARRAY,
				POOL_VECTOR2I_ARRAY,
				POOL_VECTOR3_ARRAY,
				POOL_VECTOR3I_ARRAY,
				POOL_VECTOR4_ARRAY,
				POOL_VECTOR4I_ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		// arrays
		case POOL_BYTE_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		case POOL_INT_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;
		} break;
		case POOL_REAL_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		case POOL_STRING_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		case POOL_VECTOR2_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		case POOL_VECTOR2I_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		case POOL_VECTOR3_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		case POOL_VECTOR3I_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		case POOL_VECTOR4_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		case POOL_VECTOR4I_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		case POOL_COLOR_ARRAY: {
			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		default: {
		}
	}

	if (valid_types) {
		int i = 0;
		while (valid_types[i] != NIL) {
			if (p_type_from == valid_types[i]) {
				return true;
			}
			i++;
		}
	}

	return false;
}

bool Variant::deep_equal(const Variant &p_variant, int p_recursion_count) const {
	ERR_FAIL_COND_V_MSG(p_recursion_count > MAX_RECURSION, true, "Max recursion reached");

	// Containers must be handled with recursivity checks
	switch (type) {
		case Variant::Type::DICTIONARY: {
			if (p_variant.type != Variant::Type::DICTIONARY) {
				return false;
			}

			const Dictionary v1_as_d = Dictionary(*this);
			const Dictionary v2_as_d = Dictionary(p_variant);

			return v1_as_d.deep_equal(v2_as_d, p_recursion_count + 1);
		} break;
		case Variant::Type::ARRAY: {
			if (p_variant.type != Variant::Type::ARRAY) {
				return false;
			}

			const Array v1_as_a = Array(*this);
			const Array v2_as_a = Array(p_variant);

			return v1_as_a.deep_equal(v2_as_a, p_recursion_count + 1);
		} break;
		default: {
			return *this == p_variant;
		} break;
	}
}

bool Variant::operator==(const Variant &p_variant) const {
	if (type != p_variant.type) { //evaluation of operator== needs to be more strict
		return false;
	}
	bool v;
	Variant r;
	evaluate(OP_EQUAL, *this, p_variant, r, v);
	return r;
}

bool Variant::operator!=(const Variant &p_variant) const {
	if (type != p_variant.type) { //evaluation of operator== needs to be more strict
		return true;
	}
	bool v;
	Variant r;
	evaluate(OP_NOT_EQUAL, *this, p_variant, r, v);
	return r;
}

bool Variant::operator<(const Variant &p_variant) const {
	if (type != p_variant.type) { //if types differ, then order by type first
		return type < p_variant.type;
	}
	bool v;
	Variant r;
	evaluate(OP_LESS, *this, p_variant, r, v);
	return r;
}

bool Variant::is_zero() const {
	switch (type) {
		case NIL: {
			return true;
		} break;

		// atomic types
		case BOOL: {
			return !(_data._bool);
		} break;
		case INT: {
			return _data._int == 0;
		} break;
		case REAL: {
			return _data._real == 0;
		} break;
		case STRING: {
			return *reinterpret_cast<const String *>(_data._mem) == String();
		} break;

		// math types
		case RECT2: {
			return *reinterpret_cast<const Rect2 *>(_data._mem) == Rect2();
		} break;
		case RECT2I: {
			return *reinterpret_cast<const Rect2i *>(_data._mem) == Rect2i();
		} break;
		case VECTOR2: {
			return *reinterpret_cast<const Vector2 *>(_data._mem) == Vector2();
		} break;
		case VECTOR2I: {
			return *reinterpret_cast<const Vector2i *>(_data._mem) == Vector2i();
		} break;
		case VECTOR3: {
			return *reinterpret_cast<const Vector3 *>(_data._mem) == Vector3();
		} break;
		case VECTOR3I: {
			return *reinterpret_cast<const Vector3i *>(_data._mem) == Vector3i();
		} break;
		case VECTOR4: {
			return *reinterpret_cast<const Vector4 *>(_data._mem) == Vector4();
		} break;
		case VECTOR4I: {
			return *reinterpret_cast<const Vector4i *>(_data._mem) == Vector4i();
		} break;

		case PLANE: {
			return *reinterpret_cast<const Plane *>(_data._mem) == Plane();
		} break;
		case QUATERNION: {
			return *reinterpret_cast<const Quaternion *>(_data._mem) == Quaternion();
		} break;
		case AABB: {
			return *_data._aabb == ::AABB();
		} break;
		case BASIS: {
			return *_data._basis == Basis();
		} break;
		case TRANSFORM: {
			return *_data._transform == Transform();
		} break;
		case TRANSFORM2D: {
			return *_data._transform2d == Transform2D();
		} break;
		case PROJECTION: {
			return *_data._projection == Projection();
		} break;

		// misc types
		case COLOR: {
			return *reinterpret_cast<const Color *>(_data._mem) == Color();
		} break;
		case OBJECT: {
			return _UNSAFE_OBJ_PROXY_PTR(*this) == nullptr;
		} break;
		case STRING_NAME: {
			return *reinterpret_cast<const StringName *>(_data._mem) != StringName();
		} break;
		case DICTIONARY: {
			return reinterpret_cast<const Dictionary *>(_data._mem)->empty();
		} break;
		case ARRAY: {
			return reinterpret_cast<const Array *>(_data._mem)->empty();
		} break;

		// arrays
		case POOL_BYTE_ARRAY: {
			return reinterpret_cast<const PoolVector<uint8_t> *>(_data._mem)->size() == 0;
		} break;
		case POOL_INT_ARRAY: {
			return reinterpret_cast<const PoolVector<int> *>(_data._mem)->size() == 0;
		} break;
		case POOL_REAL_ARRAY: {
			return reinterpret_cast<const PoolVector<real_t> *>(_data._mem)->size() == 0;
		} break;
		case POOL_STRING_ARRAY: {
			return reinterpret_cast<const PoolVector<String> *>(_data._mem)->size() == 0;
		} break;
		case POOL_VECTOR2_ARRAY: {
			return reinterpret_cast<const PoolVector<Vector2> *>(_data._mem)->size() == 0;
		} break;
		case POOL_VECTOR2I_ARRAY: {
			return reinterpret_cast<const PoolVector<Vector2i> *>(_data._mem)->size() == 0;
		} break;
		case POOL_VECTOR3_ARRAY: {
			return reinterpret_cast<const PoolVector<Vector3> *>(_data._mem)->size() == 0;
		} break;
		case POOL_VECTOR3I_ARRAY: {
			return reinterpret_cast<const PoolVector<Vector3i> *>(_data._mem)->size() == 0;
		} break;
		case POOL_VECTOR4_ARRAY: {
			return reinterpret_cast<const PoolVector<Vector4> *>(_data._mem)->size() == 0;
		} break;
		case POOL_VECTOR4I_ARRAY: {
			return reinterpret_cast<const PoolVector<Vector4i> *>(_data._mem)->size() == 0;
		} break;
		case POOL_COLOR_ARRAY: {
			return reinterpret_cast<const PoolVector<Color> *>(_data._mem)->size() == 0;
		} break;
		default: {
		}
	}

	return false;
}

bool Variant::is_one() const {
	switch (type) {
		case NIL: {
			return true;
		} break;

		// atomic types
		case BOOL: {
			return _data._bool;
		} break;
		case INT: {
			return _data._int == 1;
		} break;
		case REAL: {
			return _data._real == 1;
		} break;
		case RECT2: {
			return *reinterpret_cast<const Rect2 *>(_data._mem) == Rect2(1, 1, 1, 1);
		} break;
		case RECT2I: {
			return *reinterpret_cast<const Rect2i *>(_data._mem) == Rect2i(1, 1, 1, 1);
		} break;
		case VECTOR2: {
			return *reinterpret_cast<const Vector2 *>(_data._mem) == Vector2(1, 1);
		} break;
		case VECTOR2I: {
			return *reinterpret_cast<const Vector2i *>(_data._mem) == Vector2i(1, 1);
		} break;
		case VECTOR3: {
			return *reinterpret_cast<const Vector3 *>(_data._mem) == Vector3(1, 1, 1);
		} break;
		case VECTOR3I: {
			return *reinterpret_cast<const Vector3i *>(_data._mem) == Vector3i(1, 1, 1);
		} break;
		case VECTOR4: {
			return *reinterpret_cast<const Vector4 *>(_data._mem) == Vector4(1, 1, 1, 1);
		} break;
		case VECTOR4I: {
			return *reinterpret_cast<const Vector4i *>(_data._mem) == Vector4i(1, 1, 1, 1);
		} break;
		case PLANE: {
			return *reinterpret_cast<const Plane *>(_data._mem) == Plane(1, 1, 1, 1);
		} break;
		case COLOR: {
			return *reinterpret_cast<const Color *>(_data._mem) == Color(1, 1, 1, 1);
		} break;

		default: {
			return !is_zero();
		}
	}

	return false;
}

ObjectID Variant::get_object_instance_id() const {
	if (unlikely(type != OBJECT)) {
		return 0;
	} else if (likely(_get_obj().rc)) {
		return _get_obj().rc->instance_id;
	} else if (likely(!_get_obj().ref.is_null())) {
		return _REF_OBJ_PTR(*this)->get_instance_id();
	} else {
		return 0;
	}
}

bool Variant::is_invalid_object() const {
	return type == OBJECT && _get_obj().rc && !_get_obj().rc->get_ptr();
}

void Variant::reference(const Variant &p_variant) {
	switch (type) {
		case NIL:
		case BOOL:
		case INT:
		case REAL:
			break;
		default:
			clear();
	}

	type = p_variant.type;

	switch (p_variant.type) {
		case NIL: {
			// none
		} break;

		// atomic types
		case BOOL: {
			_data._bool = p_variant._data._bool;
		} break;
		case INT: {
			_data._int = p_variant._data._int;
		} break;
		case REAL: {
			_data._real = p_variant._data._real;
		} break;
		case STRING: {
			memnew_placement(_data._mem, String(*reinterpret_cast<const String *>(p_variant._data._mem)));
		} break;

		// math types
		case RECT2: {
			memnew_placement(_data._mem, Rect2(*reinterpret_cast<const Rect2 *>(p_variant._data._mem)));
		} break;
		case RECT2I: {
			memnew_placement(_data._mem, Rect2i(*reinterpret_cast<const Rect2i *>(p_variant._data._mem)));
		} break;
		case VECTOR2: {
			memnew_placement(_data._mem, Vector2(*reinterpret_cast<const Vector2 *>(p_variant._data._mem)));
		} break;
		case VECTOR2I: {
			memnew_placement(_data._mem, Vector2i(*reinterpret_cast<const Vector2i *>(p_variant._data._mem)));
		} break;
		case VECTOR3: {
			memnew_placement(_data._mem, Vector3(*reinterpret_cast<const Vector3 *>(p_variant._data._mem)));
		} break;
		case VECTOR3I: {
			memnew_placement(_data._mem, Vector3i(*reinterpret_cast<const Vector3i *>(p_variant._data._mem)));
		} break;
		case VECTOR4: {
			memnew_placement(_data._mem, Vector4(*reinterpret_cast<const Vector4 *>(p_variant._data._mem)));
		} break;
		case VECTOR4I: {
			memnew_placement(_data._mem, Vector4i(*reinterpret_cast<const Vector4i *>(p_variant._data._mem)));
		} break;

		case PLANE: {
			memnew_placement(_data._mem, Plane(*reinterpret_cast<const Plane *>(p_variant._data._mem)));
		} break;
		case QUATERNION: {
			memnew_placement(_data._mem, Quaternion(*reinterpret_cast<const Quaternion *>(p_variant._data._mem)));
		} break;
		case AABB: {
			_data._aabb = memnew(::AABB(*p_variant._data._aabb));
		} break;
		case BASIS: {
			_data._basis = memnew(Basis(*p_variant._data._basis));
		} break;
		case TRANSFORM: {
			_data._transform = memnew(Transform(*p_variant._data._transform));
		} break;
		case TRANSFORM2D: {
			_data._transform2d = memnew(Transform2D(*p_variant._data._transform2d));
		} break;
		case PROJECTION: {
			_data._projection = memnew(Projection(*p_variant._data._projection));
		} break;

		// misc types
		case COLOR: {
			memnew_placement(_data._mem, Color(*reinterpret_cast<const Color *>(p_variant._data._mem)));
		} break;
		case OBJECT: {
			memnew_placement(_data._mem, ObjData(p_variant._get_obj()));
			if (likely(_get_obj().rc)) {
				_get_obj().rc->increment();
			}
		} break;
		case STRING_NAME: {
			memnew_placement(_data._mem, StringName(*reinterpret_cast<const StringName *>(p_variant._data._mem)));
		} break;
		case DICTIONARY: {
			memnew_placement(_data._mem, Dictionary(*reinterpret_cast<const Dictionary *>(p_variant._data._mem)));
		} break;
		case ARRAY: {
			memnew_placement(_data._mem, Array(*reinterpret_cast<const Array *>(p_variant._data._mem)));
		} break;

		// arrays
		case POOL_BYTE_ARRAY: {
			memnew_placement(_data._mem, PoolVector<uint8_t>(*reinterpret_cast<const PoolVector<uint8_t> *>(p_variant._data._mem)));
		} break;
		case POOL_INT_ARRAY: {
			memnew_placement(_data._mem, PoolVector<int>(*reinterpret_cast<const PoolVector<int> *>(p_variant._data._mem)));
		} break;
		case POOL_REAL_ARRAY: {
			memnew_placement(_data._mem, PoolVector<real_t>(*reinterpret_cast<const PoolVector<real_t> *>(p_variant._data._mem)));
		} break;
		case POOL_STRING_ARRAY: {
			memnew_placement(_data._mem, PoolVector<String>(*reinterpret_cast<const PoolVector<String> *>(p_variant._data._mem)));
		} break;
		case POOL_VECTOR2_ARRAY: {
			memnew_placement(_data._mem, PoolVector<Vector2>(*reinterpret_cast<const PoolVector<Vector2> *>(p_variant._data._mem)));
		} break;
		case POOL_VECTOR2I_ARRAY: {
			memnew_placement(_data._mem, PoolVector<Vector2i>(*reinterpret_cast<const PoolVector<Vector2i> *>(p_variant._data._mem)));
		} break;
		case POOL_VECTOR3_ARRAY: {
			memnew_placement(_data._mem, PoolVector<Vector3>(*reinterpret_cast<const PoolVector<Vector3> *>(p_variant._data._mem)));
		} break;
		case POOL_VECTOR3I_ARRAY: {
			memnew_placement(_data._mem, PoolVector<Vector3i>(*reinterpret_cast<const PoolVector<Vector3i> *>(p_variant._data._mem)));
		} break;
		case POOL_VECTOR4_ARRAY: {
			memnew_placement(_data._mem, PoolVector<Vector4>(*reinterpret_cast<const PoolVector<Vector4> *>(p_variant._data._mem)));
		} break;
		case POOL_VECTOR4I_ARRAY: {
			memnew_placement(_data._mem, PoolVector<Vector4i>(*reinterpret_cast<const PoolVector<Vector4i> *>(p_variant._data._mem)));
		} break;
		case POOL_COLOR_ARRAY: {
			memnew_placement(_data._mem, PoolVector<Color>(*reinterpret_cast<const PoolVector<Color> *>(p_variant._data._mem)));
		} break;
		default: {
		}
	}
}

void Variant::zero() {
	switch (type) {
		case NIL:
			break;
		case BOOL:
			this->_data._bool = false;
			break;
		case INT:
			this->_data._int = 0;
			break;
		case REAL:
			this->_data._real = 0;
			break;
		case RECT2:
			*reinterpret_cast<Rect2 *>(this->_data._mem) = Rect2();
			break;
		case RECT2I:
			*reinterpret_cast<Rect2i *>(this->_data._mem) = Rect2i();
			break;
		case VECTOR2:
			*reinterpret_cast<Vector2 *>(this->_data._mem) = Vector2();
			break;
		case VECTOR2I:
			*reinterpret_cast<Vector2i *>(this->_data._mem) = Vector2i();
			break;
		case VECTOR3:
			*reinterpret_cast<Vector3 *>(this->_data._mem) = Vector3();
			break;
		case VECTOR3I:
			*reinterpret_cast<Vector3i *>(this->_data._mem) = Vector3i();
			break;
		case VECTOR4:
			*reinterpret_cast<Vector4 *>(this->_data._mem) = Vector4();
			break;
		case VECTOR4I:
			*reinterpret_cast<Vector4i *>(this->_data._mem) = Vector4i();
			break;
		case PLANE:
			*reinterpret_cast<Plane *>(this->_data._mem) = Plane();
			break;
		case QUATERNION:
			*reinterpret_cast<Quaternion *>(this->_data._mem) = Quaternion();
			break;
		case AABB:
			*reinterpret_cast<::AABB *>(this->_data._mem) = ::AABB();
			break;
		case COLOR:
			*reinterpret_cast<Color *>(this->_data._mem) = Color();
			break;
		case PROJECTION:
			*reinterpret_cast<Projection *>(this->_data._mem) = Projection();
			break;
		default:
			this->clear();
			break;
	}
}

void Variant::clear() {
	switch (type) {
		/*
		// no point, they don't allocate memory
		BOOL,
		INT,
		REAL,
		*/
		case STRING: {
			reinterpret_cast<String *>(_data._mem)->~String();
		} break;
		/*
		RECT2,
		RECT2I
		VECTOR2,
		VECTOR2I,
		VECTOR3,
		VECTOR3i,
		VECTOR4,
		VECTOR4i,

		PLANE,
		QUATERNION,
		*/
		case AABB: {
			memdelete(_data._aabb);
		} break;
		case BASIS: {
			memdelete(_data._basis);
		} break;
		case TRANSFORM: {
			memdelete(_data._transform);
		} break;
		case TRANSFORM2D: {
			memdelete(_data._transform2d);
		} break;
		case PROJECTION: {
			memdelete(_data._projection);
		} break;
		//COLOR

		// misc types
		case OBJECT: {
			if (likely(_get_obj().rc)) {
				if (unlikely(_get_obj().rc->decrement())) {
					memdelete(_get_obj().rc);
				}
			} else {
				_get_obj().ref.unref();
			}
		} break;
		case STRING_NAME: {
			reinterpret_cast<StringName *>(_data._mem)->~StringName();
		} break;
		case DICTIONARY: {
			reinterpret_cast<Dictionary *>(_data._mem)->~Dictionary();
		} break;
		case ARRAY: {
			reinterpret_cast<Array *>(_data._mem)->~Array();
		} break;

		// arrays
		case POOL_BYTE_ARRAY: {
			reinterpret_cast<PoolVector<uint8_t> *>(_data._mem)->~PoolVector<uint8_t>();
		} break;
		case POOL_INT_ARRAY: {
			reinterpret_cast<PoolVector<int> *>(_data._mem)->~PoolVector<int>();
		} break;
		case POOL_REAL_ARRAY: {
			reinterpret_cast<PoolVector<real_t> *>(_data._mem)->~PoolVector<real_t>();
		} break;
		case POOL_STRING_ARRAY: {
			reinterpret_cast<PoolVector<String> *>(_data._mem)->~PoolVector<String>();
		} break;
		case POOL_VECTOR2_ARRAY: {
			reinterpret_cast<PoolVector<Vector2> *>(_data._mem)->~PoolVector<Vector2>();
		} break;
		case POOL_VECTOR2I_ARRAY: {
			reinterpret_cast<PoolVector<Vector2i> *>(_data._mem)->~PoolVector<Vector2i>();
		} break;
		case POOL_VECTOR3_ARRAY: {
			reinterpret_cast<PoolVector<Vector3> *>(_data._mem)->~PoolVector<Vector3>();
		} break;
		case POOL_VECTOR3I_ARRAY: {
			reinterpret_cast<PoolVector<Vector3i> *>(_data._mem)->~PoolVector<Vector3i>();
		} break;
		case POOL_VECTOR4_ARRAY: {
			reinterpret_cast<PoolVector<Vector4> *>(_data._mem)->~PoolVector<Vector4>();
		} break;
		case POOL_VECTOR4I_ARRAY: {
			reinterpret_cast<PoolVector<Vector4i> *>(_data._mem)->~PoolVector<Vector4i>();
		} break;
		case POOL_COLOR_ARRAY: {
			reinterpret_cast<PoolVector<Color> *>(_data._mem)->~PoolVector<Color>();
		} break;
		default: {
		} /* not needed */
	}

	type = NIL;
}

Variant::operator signed int() const {
	switch (type) {
		case NIL:
			return 0;
		case BOOL:
			return _data._bool ? 1 : 0;
		case INT:
			return _data._int;
		case REAL:
			return _data._real;
		case STRING:
			return operator String().to_int();
		default: {
			return 0;
		}
	}
}
Variant::operator unsigned int() const {
	switch (type) {
		case NIL:
			return 0;
		case BOOL:
			return _data._bool ? 1 : 0;
		case INT:
			return _data._int;
		case REAL:
			return _data._real;
		case STRING:
			return operator String().to_int();
		default: {
			return 0;
		}
	}
}

Variant::operator int64_t() const {
	switch (type) {
		case NIL:
			return 0;
		case BOOL:
			return _data._bool ? 1 : 0;
		case INT:
			return _data._int;
		case REAL:
			return _data._real;
		case STRING:
			return operator String().to_int64();
		default: {
			return 0;
		}
	}
}

/*
Variant::operator long unsigned int() const {

	switch( type ) {

		case NIL: return 0;
		case BOOL: return _data._bool ? 1 : 0;
		case INT: return _data._int;
		case REAL: return _data._real;
		case STRING: return operator String().to_int();
		default: {

			return 0;
		}
	}

	return 0;
};
*/

Variant::operator uint64_t() const {
	switch (type) {
		case NIL:
			return 0;
		case BOOL:
			return _data._bool ? 1 : 0;
		case INT:
			return _data._int;
		case REAL:
			return _data._real;
		case STRING:
			return operator String().to_int();
		default: {
			return 0;
		}
	}
}

#ifdef NEED_LONG_INT
Variant::operator signed long() const {
	switch (type) {
		case NIL:
			return 0;
		case BOOL:
			return _data._bool ? 1 : 0;
		case INT:
			return _data._int;
		case REAL:
			return _data._real;
		case STRING:
			return operator String().to_int();
		default: {
			return 0;
		}
	}

	return 0;
};

Variant::operator unsigned long() const {
	switch (type) {
		case NIL:
			return 0;
		case BOOL:
			return _data._bool ? 1 : 0;
		case INT:
			return _data._int;
		case REAL:
			return _data._real;
		case STRING:
			return operator String().to_int();
		default: {
			return 0;
		}
	}

	return 0;
};
#endif

Variant::operator signed short() const {
	switch (type) {
		case NIL:
			return 0;
		case BOOL:
			return _data._bool ? 1 : 0;
		case INT:
			return _data._int;
		case REAL:
			return _data._real;
		case STRING:
			return operator String().to_int();
		default: {
			return 0;
		}
	}
}
Variant::operator unsigned short() const {
	switch (type) {
		case NIL:
			return 0;
		case BOOL:
			return _data._bool ? 1 : 0;
		case INT:
			return _data._int;
		case REAL:
			return _data._real;
		case STRING:
			return operator String().to_int();
		default: {
			return 0;
		}
	}
}
Variant::operator signed char() const {
	switch (type) {
		case NIL:
			return 0;
		case BOOL:
			return _data._bool ? 1 : 0;
		case INT:
			return _data._int;
		case REAL:
			return _data._real;
		case STRING:
			return operator String().to_int();
		default: {
			return 0;
		}
	}
}
Variant::operator unsigned char() const {
	switch (type) {
		case NIL:
			return 0;
		case BOOL:
			return _data._bool ? 1 : 0;
		case INT:
			return _data._int;
		case REAL:
			return _data._real;
		case STRING:
			return operator String().to_int();
		default: {
			return 0;
		}
	}
}

Variant::operator CharType() const {
	return operator unsigned int();
}

Variant::operator float() const {
	switch (type) {
		case NIL:
			return 0;
		case BOOL:
			return _data._bool ? 1.0 : 0.0;
		case INT:
			return (float)_data._int;
		case REAL:
			return _data._real;
		case STRING:
			return operator String().to_double();
		default: {
			return 0;
		}
	}
}
Variant::operator double() const {
	switch (type) {
		case NIL:
			return 0;
		case BOOL:
			return _data._bool ? 1.0 : 0.0;
		case INT:
			return (double)_data._int;
		case REAL:
			return _data._real;
		case STRING:
			return operator String().to_double();
		default: {
			return 0;
		}
	}
}

Variant::operator StringName() const {
	if (type == STRING_NAME) {
		return *reinterpret_cast<const StringName *>(_data._mem);
	}

	return StringName(operator String());
}

struct _VariantStrPair {
	String key;
	String value;

	bool operator<(const _VariantStrPair &p) const {
		return key < p.key;
	}
};

Variant::operator String() const {
	List<const void *> stack;

	return stringify(stack);
}

template <class T>
String stringify_vector(const T &vec, List<const void *> &stack) {
	String str("[");
	for (int i = 0; i < vec.size(); i++) {
		if (i > 0) {
			str += ", ";
		}
		str = str + Variant(vec[i]).stringify(stack);
	}
	str += "]";
	return str;
}

String Variant::stringify(List<const void *> &stack) const {
	switch (type) {
		case NIL:
			return "Null";
		case BOOL:
			return _data._bool ? "True" : "False";
		case INT:
			return itos(_data._int);
		case REAL:
			return rtos(_data._real);
		case STRING:
			return *reinterpret_cast<const String *>(_data._mem);
		case RECT2:
			return operator Rect2();
		case RECT2I:
			return operator Rect2i();
		case VECTOR2:
			return operator Vector2();
		case VECTOR2I:
			return operator Vector2i();
		case VECTOR3:
			return operator Vector3();
		case VECTOR3I:
			return operator Vector3i();
		case VECTOR4:
			return operator Vector4();
		case VECTOR4I:
			return operator Vector4i();
		case PLANE:
			return operator Plane();
		case QUATERNION:
			return operator Quaternion();
		case AABB:
			return operator ::AABB();
		case BASIS: {
			return operator Basis();
		} break;
		case TRANSFORM:
			return operator Transform();
		case TRANSFORM2D: {
			return operator Transform2D();
		} break;
		case PROJECTION:
			return operator Projection();
		case COLOR:
			return operator Color();
		case OBJECT: {
			Object *obj = _OBJ_PTR(*this);
			if (likely(obj)) {
				return obj->to_string();
			} else {
				if (_get_obj().rc) {
					return "[Deleted Object]";
				}
				return "[Object:null]";
			}
		} break;
		case STRING_NAME:
			return operator StringName();
		case DICTIONARY: {
			const Dictionary &d = *reinterpret_cast<const Dictionary *>(_data._mem);
			if (stack.find(d.id())) {
				return "{...}";
			}

			stack.push_back(d.id());

			//const String *K=NULL;
			String str("{");
			List<Variant> keys;
			d.get_key_list(&keys);

			Vector<_VariantStrPair> pairs;

			for (List<Variant>::Element *E = keys.front(); E; E = E->next()) {
				_VariantStrPair sp;
				sp.key = E->get().stringify(stack);
				sp.value = d[E->get()].stringify(stack);

				pairs.push_back(sp);
			}

			pairs.sort();

			for (int i = 0; i < pairs.size(); i++) {
				if (i > 0) {
					str += ", ";
				}
				str += pairs[i].key + ":" + pairs[i].value;
			}
			str += "}";

			stack.erase(d.id());
			return str;
		} break;
		case ARRAY: {
			Array arr = operator Array();
			if (stack.find(arr.id())) {
				return "[...]";
			}
			stack.push_back(arr.id());
			String str = stringify_vector(arr, stack);
			stack.erase(arr.id());
			return str;

		} break;

		case POOL_BYTE_ARRAY: {
			return stringify_vector(operator PoolVector<uint8_t>(), stack);
		} break;
		case POOL_INT_ARRAY: {
			return stringify_vector(operator PoolVector<int>(), stack);
		} break;
		case POOL_REAL_ARRAY: {
			return stringify_vector(operator PoolVector<real_t>(), stack);
		} break;
		case POOL_STRING_ARRAY: {
			return stringify_vector(operator PoolVector<String>(), stack);
		} break;
		case POOL_VECTOR2_ARRAY: {
			return stringify_vector(operator PoolVector<Vector2>(), stack);
		} break;
		case POOL_VECTOR2I_ARRAY: {
			return stringify_vector(operator PoolVector<Vector2i>(), stack);
		} break;
		case POOL_VECTOR3_ARRAY: {
			return stringify_vector(operator PoolVector<Vector3>(), stack);
		} break;
		case POOL_VECTOR3I_ARRAY: {
			return stringify_vector(operator PoolVector<Vector3i>(), stack);
		} break;
		case POOL_VECTOR4_ARRAY: {
			return stringify_vector(operator PoolVector<Vector4>(), stack);
		} break;
		case POOL_VECTOR4I_ARRAY: {
			return stringify_vector(operator PoolVector<Vector4i>(), stack);
		} break;
		case POOL_COLOR_ARRAY: {
			return stringify_vector(operator PoolVector<Color>(), stack);
		} break;

		default: {
			return "[" + get_type_name(type) + "]";
		}
	}

	return "";
}

Variant::operator Rect2() const {
	if (type == RECT2) {
		return *reinterpret_cast<const Rect2 *>(_data._mem);
	} else if (type == RECT2I) {
		return Rect2(*reinterpret_cast<const Rect2i *>(_data._mem));
	} else {
		return Rect2();
	}
}
Variant::operator Rect2i() const {
	if (type == RECT2I) {
		return *reinterpret_cast<const Rect2i *>(_data._mem);
	} else if (type == RECT2) {
		return Rect2i(*reinterpret_cast<const Rect2 *>(_data._mem));
	} else {
		return Rect2i();
	}
}

Variant::operator Vector2() const {
	if (type == VECTOR2) {
		return *reinterpret_cast<const Vector2 *>(_data._mem);
	} else if (type == VECTOR2I) {
		return Vector2(reinterpret_cast<const Vector2i *>(_data._mem)->x, reinterpret_cast<const Vector2i *>(_data._mem)->y);
	} else if (type == VECTOR3) {
		return Vector2(reinterpret_cast<const Vector3 *>(_data._mem)->x, reinterpret_cast<const Vector3 *>(_data._mem)->y);
	} else if (type == VECTOR3I) {
		return Vector2(reinterpret_cast<const Vector3i *>(_data._mem)->x, reinterpret_cast<const Vector3i *>(_data._mem)->y);
	} else if (type == VECTOR4) {
		return Vector2(reinterpret_cast<const Vector4 *>(_data._mem)->x, reinterpret_cast<const Vector4 *>(_data._mem)->y);
	} else if (type == VECTOR4I) {
		return Vector2(reinterpret_cast<const Vector4i *>(_data._mem)->x, reinterpret_cast<const Vector4i *>(_data._mem)->y);
	} else {
		return Vector2();
	}
}
Variant::operator Vector2i() const {
	if (type == VECTOR2I) {
		return *reinterpret_cast<const Vector2i *>(_data._mem);
	} else if (type == VECTOR2) {
		return Vector2i(reinterpret_cast<const Vector2 *>(_data._mem)->x, reinterpret_cast<const Vector2 *>(_data._mem)->y);
	} else if (type == VECTOR3) {
		return Vector2i(reinterpret_cast<const Vector3 *>(_data._mem)->x, reinterpret_cast<const Vector3 *>(_data._mem)->y);
	} else if (type == VECTOR3I) {
		return Vector2i(reinterpret_cast<const Vector3i *>(_data._mem)->x, reinterpret_cast<const Vector3i *>(_data._mem)->y);
	} else if (type == VECTOR4) {
		return Vector2i(reinterpret_cast<const Vector4 *>(_data._mem)->x, reinterpret_cast<const Vector4 *>(_data._mem)->y);
	} else if (type == VECTOR4I) {
		return Vector2i(reinterpret_cast<const Vector4i *>(_data._mem)->x, reinterpret_cast<const Vector4i *>(_data._mem)->y);
	} else {
		return Vector2i();
	}
}

Variant::operator Vector3() const {
	if (type == VECTOR3) {
		return *reinterpret_cast<const Vector3 *>(_data._mem);
	} else if (type == VECTOR3I) {
		return Vector3(*reinterpret_cast<const Vector3i *>(_data._mem));
	} else if (type == VECTOR2) {
		return Vector3(reinterpret_cast<const Vector2 *>(_data._mem)->x, reinterpret_cast<const Vector2 *>(_data._mem)->y, 0.0);
	} else if (type == VECTOR2I) {
		return Vector3(reinterpret_cast<const Vector2i *>(_data._mem)->x, reinterpret_cast<const Vector2i *>(_data._mem)->y, 0.0);
	} else if (type == VECTOR4) {
		return Vector3(reinterpret_cast<const Vector4 *>(_data._mem)->x, reinterpret_cast<const Vector4 *>(_data._mem)->y, reinterpret_cast<const Vector4 *>(_data._mem)->z);
	} else if (type == VECTOR4I) {
		return Vector3(reinterpret_cast<const Vector4i *>(_data._mem)->x, reinterpret_cast<const Vector4i *>(_data._mem)->y, reinterpret_cast<const Vector4i *>(_data._mem)->z);
	} else {
		return Vector3();
	}
}
Variant::operator Vector3i() const {
	if (type == VECTOR3I) {
		return *reinterpret_cast<const Vector3i *>(_data._mem);
	} else if (type == VECTOR3) {
		return Vector3i(reinterpret_cast<const Vector3 *>(_data._mem)->x, reinterpret_cast<const Vector3 *>(_data._mem)->y, reinterpret_cast<const Vector3 *>(_data._mem)->z);
	} else if (type == VECTOR2) {
		return Vector3i(reinterpret_cast<const Vector2 *>(_data._mem)->x, reinterpret_cast<const Vector2 *>(_data._mem)->y, 0.0);
	} else if (type == VECTOR2I) {
		return Vector3i(reinterpret_cast<const Vector2i *>(_data._mem)->x, reinterpret_cast<const Vector2i *>(_data._mem)->y, 0.0);
	} else if (type == VECTOR4) {
		return Vector3i(reinterpret_cast<const Vector4 *>(_data._mem)->x, reinterpret_cast<const Vector4 *>(_data._mem)->y, reinterpret_cast<const Vector4 *>(_data._mem)->z);
	} else if (type == VECTOR4I) {
		return Vector3i(reinterpret_cast<const Vector4i *>(_data._mem)->x, reinterpret_cast<const Vector4i *>(_data._mem)->y, reinterpret_cast<const Vector4i *>(_data._mem)->z);
	} else {
		return Vector3i();
	}
}

Variant::operator Vector4() const {
	if (type == VECTOR4) {
		return *reinterpret_cast<const Vector4 *>(_data._mem);
	} else if (type == VECTOR4I) {
		return *reinterpret_cast<const Vector4i *>(_data._mem);
	} else if (type == VECTOR2) {
		return Vector4(reinterpret_cast<const Vector2 *>(_data._mem)->x, reinterpret_cast<const Vector2 *>(_data._mem)->y, 0.0, 0.0);
	} else if (type == VECTOR2I) {
		return Vector4(reinterpret_cast<const Vector2i *>(_data._mem)->x, reinterpret_cast<const Vector2i *>(_data._mem)->y, 0.0, 0.0);
	} else if (type == VECTOR3) {
		return Vector4(reinterpret_cast<const Vector3 *>(_data._mem)->x, reinterpret_cast<const Vector3 *>(_data._mem)->y, reinterpret_cast<const Vector3 *>(_data._mem)->z, 0.0);
	} else if (type == VECTOR3I) {
		return Vector4(reinterpret_cast<const Vector3i *>(_data._mem)->x, reinterpret_cast<const Vector3i *>(_data._mem)->y, reinterpret_cast<const Vector3i *>(_data._mem)->z, 0.0);
	} else {
		return Vector4();
	}
}

Variant::operator Vector4i() const {
	if (type == VECTOR4I) {
		return *reinterpret_cast<const Vector4i *>(_data._mem);
	} else if (type == VECTOR4) {
		const Vector4 &v4 = *reinterpret_cast<const Vector4 *>(_data._mem);
		return Vector4i(v4.x, v4.y, v4.z, v4.w);
	} else if (type == VECTOR2) {
		return Vector4i(reinterpret_cast<const Vector2 *>(_data._mem)->x, reinterpret_cast<const Vector2 *>(_data._mem)->y, 0.0, 0.0);
	} else if (type == VECTOR2I) {
		return Vector4i(reinterpret_cast<const Vector2i *>(_data._mem)->x, reinterpret_cast<const Vector2i *>(_data._mem)->y, 0.0, 0.0);
	} else if (type == VECTOR3) {
		return Vector4i(reinterpret_cast<const Vector3 *>(_data._mem)->x, reinterpret_cast<const Vector3 *>(_data._mem)->y, reinterpret_cast<const Vector3 *>(_data._mem)->z, 0.0);
	} else if (type == VECTOR3I) {
		return Vector4i(reinterpret_cast<const Vector3i *>(_data._mem)->x, reinterpret_cast<const Vector3i *>(_data._mem)->y, reinterpret_cast<const Vector3i *>(_data._mem)->z, 0.0);
	} else {
		return Vector4i();
	}
}

Variant::operator Plane() const {
	if (type == PLANE) {
		return *reinterpret_cast<const Plane *>(_data._mem);
	} else {
		return Plane();
	}
}
Variant::operator ::AABB() const {
	if (type == AABB) {
		return *_data._aabb;
	} else {
		return ::AABB();
	}
}

Variant::operator Basis() const {
	if (type == BASIS) {
		return *_data._basis;
	} else if (type == QUATERNION) {
		return *reinterpret_cast<const Quaternion *>(_data._mem);
	} else if (type == VECTOR3) {
		return Basis(*reinterpret_cast<const Vector3 *>(_data._mem));
	} else if (type == TRANSFORM) { // unexposed in Variant::can_convert?
		return _data._transform->basis;
	} else {
		return Basis();
	}
}

Variant::operator Quaternion() const {
	if (type == QUATERNION) {
		return *reinterpret_cast<const Quaternion *>(_data._mem);
	} else if (type == BASIS) {
		return *_data._basis;
	} else if (type == TRANSFORM) {
		return _data._transform->basis;
	} else {
		return Quaternion();
	}
}

Variant::operator Transform2D() const {
	if (type == TRANSFORM2D) {
		return *_data._transform2d;
	} else if (type == TRANSFORM) {
		const Transform &t = *_data._transform;
		Transform2D m;
		m.columns[0][0] = t.basis.rows[0][0];
		m.columns[0][1] = t.basis.rows[1][0];
		m.columns[1][0] = t.basis.rows[0][1];
		m.columns[1][1] = t.basis.rows[1][1];
		m.columns[2][0] = t.origin[0];
		m.columns[2][1] = t.origin[1];
		return m;
	} else {
		return Transform2D();
	}
}

Variant::operator Transform() const {
	if (type == TRANSFORM) {
		return *_data._transform;
	} else if (type == BASIS) {
		return Transform(*_data._basis, Vector3());
	} else if (type == QUATERNION) {
		return Transform(Basis(*reinterpret_cast<const Quaternion *>(_data._mem)), Vector3());
	} else if (type == TRANSFORM2D) {
		const Transform2D &t = *_data._transform2d;
		Transform m;
		m.basis.rows[0][0] = t.columns[0][0];
		m.basis.rows[1][0] = t.columns[0][1];
		m.basis.rows[0][1] = t.columns[1][0];
		m.basis.rows[1][1] = t.columns[1][1];
		m.origin[0] = t.columns[2][0];
		m.origin[1] = t.columns[2][1];
		return m;
	} else if (type == PROJECTION) {
		return *_data._projection;
	} else {
		return Transform();
	}
}

Variant::operator Projection() const {
	if (type == TRANSFORM) {
		return *_data._transform;
	} else if (type == BASIS) {
		return Transform(*_data._basis, Vector3());
	} else if (type == QUATERNION) {
		return Transform(Basis(*reinterpret_cast<const Quaternion *>(_data._mem)), Vector3());
	} else if (type == TRANSFORM2D) {
		const Transform2D &t = *_data._transform2d;
		Transform m;
		m.basis.rows[0][0] = t.columns[0][0];
		m.basis.rows[1][0] = t.columns[0][1];
		m.basis.rows[0][1] = t.columns[1][0];
		m.basis.rows[1][1] = t.columns[1][1];
		m.origin[0] = t.columns[2][0];
		m.origin[1] = t.columns[2][1];
		return m;
	} else if (type == PROJECTION) {
		return *_data._projection;
	} else {
		return Projection();
	}
}

Variant::operator Color() const {
	if (type == COLOR) {
		return *reinterpret_cast<const Color *>(_data._mem);
	} else if (type == STRING) {
		return Color::html(operator String());
	} else if (type == INT) {
		return Color::hex(operator int());
	} else {
		return Color();
	}
}

Variant::operator RefPtr() const {
	if (type == OBJECT) {
		return _get_obj().ref;
	} else {
		return RefPtr();
	}
}

Variant::operator Object *() const {
	if (type == OBJECT) {
		return _OBJ_PTR(*this);
	} else {
		return nullptr;
	}
}

template <class DA, class SA>
inline DA _convert_array(const SA &p_array) {
	DA da;
	da.resize(p_array.size());

	for (int i = 0; i < p_array.size(); i++) {
		da.set(i, Variant(p_array.get(i)));
	}

	return da;
}

template <class DA>
inline DA _convert_array_from_variant(const Variant &p_variant) {
	switch (p_variant.get_type()) {
		case Variant::ARRAY: {
			return _convert_array<DA, Array>(p_variant.operator Array());
		}
		case Variant::POOL_BYTE_ARRAY: {
			return _convert_array<DA, PoolVector<uint8_t>>(p_variant.operator PoolVector<uint8_t>());
		}
		case Variant::POOL_INT_ARRAY: {
			return _convert_array<DA, PoolVector<int>>(p_variant.operator PoolVector<int>());
		}
		case Variant::POOL_REAL_ARRAY: {
			return _convert_array<DA, PoolVector<real_t>>(p_variant.operator PoolVector<real_t>());
		}
		case Variant::POOL_STRING_ARRAY: {
			return _convert_array<DA, PoolVector<String>>(p_variant.operator PoolVector<String>());
		}
		case Variant::POOL_VECTOR2_ARRAY: {
			return _convert_array<DA, PoolVector<Vector2>>(p_variant.operator PoolVector<Vector2>());
		}
		case Variant::POOL_VECTOR2I_ARRAY: {
			return _convert_array<DA, PoolVector<Vector2i>>(p_variant.operator PoolVector<Vector2i>());
		}
		case Variant::POOL_VECTOR3_ARRAY: {
			return _convert_array<DA, PoolVector<Vector3>>(p_variant.operator PoolVector<Vector3>());
		}
		case Variant::POOL_VECTOR3I_ARRAY: {
			return _convert_array<DA, PoolVector<Vector3i>>(p_variant.operator PoolVector<Vector3i>());
		}
		case Variant::POOL_VECTOR4_ARRAY: {
			return _convert_array<DA, PoolVector<Vector4>>(p_variant.operator PoolVector<Vector4>());
		}
		case Variant::POOL_VECTOR4I_ARRAY: {
			return _convert_array<DA, PoolVector<Vector4i>>(p_variant.operator PoolVector<Vector4i>());
		}
		case Variant::POOL_COLOR_ARRAY: {
			return _convert_array<DA, PoolVector<Color>>(p_variant.operator PoolVector<Color>());
		}
		default: {
			return DA();
		}
	}
}

Variant::operator Dictionary() const {
	if (type == DICTIONARY) {
		return *reinterpret_cast<const Dictionary *>(_data._mem);
	} else {
		return Dictionary();
	}
}

Variant::operator Array() const {
	if (type == ARRAY) {
		return *reinterpret_cast<const Array *>(_data._mem);
	} else {
		return _convert_array_from_variant<Array>(*this);
	}
}

Variant::operator PoolVector<uint8_t>() const {
	if (type == POOL_BYTE_ARRAY) {
		return *reinterpret_cast<const PoolVector<uint8_t> *>(_data._mem);
	} else {
		return _convert_array_from_variant<PoolVector<uint8_t>>(*this);
	}
}
Variant::operator PoolVector<int>() const {
	if (type == POOL_INT_ARRAY) {
		return *reinterpret_cast<const PoolVector<int> *>(_data._mem);
	} else {
		return _convert_array_from_variant<PoolVector<int>>(*this);
	}
}
Variant::operator PoolVector<real_t>() const {
	if (type == POOL_REAL_ARRAY) {
		return *reinterpret_cast<const PoolVector<real_t> *>(_data._mem);
	} else {
		return _convert_array_from_variant<PoolVector<real_t>>(*this);
	}
}

Variant::operator PoolVector<String>() const {
	if (type == POOL_STRING_ARRAY) {
		return *reinterpret_cast<const PoolVector<String> *>(_data._mem);
	} else {
		return _convert_array_from_variant<PoolVector<String>>(*this);
	}
}
Variant::operator PoolVector<Vector2>() const {
	if (type == POOL_VECTOR2_ARRAY) {
		return *reinterpret_cast<const PoolVector<Vector2> *>(_data._mem);
	} else {
		return _convert_array_from_variant<PoolVector<Vector2>>(*this);
	}
}
Variant::operator PoolVector<Vector2i>() const {
	if (type == POOL_VECTOR2I_ARRAY) {
		return *reinterpret_cast<const PoolVector<Vector2i> *>(_data._mem);
	} else {
		return _convert_array_from_variant<PoolVector<Vector2i>>(*this);
	}
}
Variant::operator PoolVector<Vector3>() const {
	if (type == POOL_VECTOR3_ARRAY) {
		return *reinterpret_cast<const PoolVector<Vector3> *>(_data._mem);
	} else {
		return _convert_array_from_variant<PoolVector<Vector3>>(*this);
	}
}
Variant::operator PoolVector<Vector3i>() const {
	if (type == POOL_VECTOR3I_ARRAY) {
		return *reinterpret_cast<const PoolVector<Vector3i> *>(_data._mem);
	} else {
		return _convert_array_from_variant<PoolVector<Vector3i>>(*this);
	}
}
Variant::operator PoolVector<Vector4>() const {
	if (type == POOL_VECTOR4_ARRAY) {
		return *reinterpret_cast<const PoolVector<Vector4> *>(_data._mem);
	} else {
		return _convert_array_from_variant<PoolVector<Vector4>>(*this);
	}
}
Variant::operator PoolVector<Vector4i>() const {
	if (type == POOL_VECTOR4I_ARRAY) {
		return *reinterpret_cast<const PoolVector<Vector4i> *>(_data._mem);
	} else {
		return _convert_array_from_variant<PoolVector<Vector4i>>(*this);
	}
}
Variant::operator PoolVector<Color>() const {
	if (type == POOL_COLOR_ARRAY) {
		return *reinterpret_cast<const PoolVector<Color> *>(_data._mem);
	} else {
		return _convert_array_from_variant<PoolVector<Color>>(*this);
	}
}

/* helpers */

Variant::operator PoolVector<Plane>() const {
	Array va = operator Array();
	PoolVector<Plane> planes;
	int va_size = va.size();
	if (va_size == 0) {
		return planes;
	}

	planes.resize(va_size);
	PoolVector<Plane>::Write w = planes.write();

	for (int i = 0; i < va_size; i++) {
		w[i] = va[i];
	}

	return planes;
}

Variant::operator PoolVector<Face3>() const {
	PoolVector<Vector3> va = operator PoolVector<Vector3>();
	PoolVector<Face3> faces;
	int va_size = va.size();
	if (va_size == 0) {
		return faces;
	}

	faces.resize(va_size / 3);
	PoolVector<Face3>::Write w = faces.write();
	PoolVector<Vector3>::Read r = va.read();

	for (int i = 0; i < va_size; i++) {
		w[i / 3].vertex[i % 3] = r[i];
	}

	return faces;
}

Variant::operator Vector<Plane>() const {
	Array va = operator Array();
	Vector<Plane> planes;
	int va_size = va.size();
	if (va_size == 0) {
		return planes;
	}

	planes.resize(va_size);

	for (int i = 0; i < va_size; i++) {
		planes.write[i] = va[i];
	}

	return planes;
}

Variant::operator Vector<Variant>() const {
	Array from = operator Array();
	Vector<Variant> to;
	int len = from.size();
	to.resize(len);
	for (int i = 0; i < len; i++) {
		to.write[i] = from[i];
	}
	return to;
}

Variant::operator Vector<uint8_t>() const {
	PoolVector<uint8_t> from = operator PoolVector<uint8_t>();
	Vector<uint8_t> to;
	int len = from.size();
	to.resize(len);
	for (int i = 0; i < len; i++) {
		to.write[i] = from[i];
	}
	return to;
}
Variant::operator Vector<int>() const {
	PoolVector<int> from = operator PoolVector<int>();
	Vector<int> to;
	int len = from.size();
	to.resize(len);
	for (int i = 0; i < len; i++) {
		to.write[i] = from[i];
	}
	return to;
}
Variant::operator Vector<real_t>() const {
	PoolVector<real_t> from = operator PoolVector<real_t>();
	Vector<real_t> to;
	int len = from.size();
	to.resize(len);
	for (int i = 0; i < len; i++) {
		to.write[i] = from[i];
	}
	return to;
}

Variant::operator Vector<String>() const {
	PoolVector<String> from = operator PoolVector<String>();
	Vector<String> to;
	int len = from.size();
	to.resize(len);
	for (int i = 0; i < len; i++) {
		to.write[i] = from[i];
	}
	return to;
}
Variant::operator Vector<StringName>() const {
	PoolVector<String> from = operator PoolVector<String>();
	Vector<StringName> to;
	int len = from.size();
	to.resize(len);
	for (int i = 0; i < len; i++) {
		to.write[i] = from[i];
	}
	return to;
}

Variant::operator Vector<Vector2>() const {
	PoolVector<Vector2> from = operator PoolVector<Vector2>();
	Vector<Vector2> to;
	int len = from.size();
	if (len == 0) {
		return Vector<Vector2>();
	}
	to.resize(len);
	PoolVector<Vector2>::Read r = from.read();
	Vector2 *w = to.ptrw();
	for (int i = 0; i < len; i++) {
		w[i] = r[i];
	}
	return to;
}
Variant::operator Vector<Vector2i>() const {
	PoolVector<Vector2i> from = operator PoolVector<Vector2i>();
	Vector<Vector2i> to;
	int len = from.size();
	if (len == 0) {
		return Vector<Vector2i>();
	}
	to.resize(len);
	PoolVector<Vector2i>::Read r = from.read();
	Vector2i *w = to.ptrw();
	for (int i = 0; i < len; i++) {
		w[i] = r[i];
	}
	return to;
}

Variant::operator Vector<Vector3>() const {
	PoolVector<Vector3> from = operator PoolVector<Vector3>();
	Vector<Vector3> to;
	int len = from.size();
	if (len == 0) {
		return Vector<Vector3>();
	}
	to.resize(len);
	PoolVector<Vector3>::Read r = from.read();
	Vector3 *w = to.ptrw();
	for (int i = 0; i < len; i++) {
		w[i] = r[i];
	}
	return to;
}
Variant::operator Vector<Vector3i>() const {
	PoolVector<Vector3i> from = operator PoolVector<Vector3i>();
	Vector<Vector3i> to;
	int len = from.size();
	if (len == 0) {
		return Vector<Vector3i>();
	}
	to.resize(len);
	PoolVector<Vector3i>::Read r = from.read();
	Vector3i *w = to.ptrw();
	for (int i = 0; i < len; i++) {
		w[i] = r[i];
	}
	return to;
}

Variant::operator Vector<Vector4>() const {
	PoolVector<Vector4> from = operator PoolVector<Vector4>();
	Vector<Vector4> to;
	int len = from.size();
	if (len == 0) {
		return Vector<Vector4>();
	}
	to.resize(len);
	PoolVector<Vector4>::Read r = from.read();
	Vector4 *w = to.ptrw();
	for (int i = 0; i < len; i++) {
		w[i] = r[i];
	}
	return to;
}
Variant::operator Vector<Vector4i>() const {
	PoolVector<Vector4i> from = operator PoolVector<Vector4i>();
	Vector<Vector4i> to;
	int len = from.size();
	if (len == 0) {
		return Vector<Vector4i>();
	}
	to.resize(len);
	PoolVector<Vector4i>::Read r = from.read();
	Vector4i *w = to.ptrw();
	for (int i = 0; i < len; i++) {
		w[i] = r[i];
	}
	return to;
}

Variant::operator Vector<Color>() const {
	PoolVector<Color> from = operator PoolVector<Color>();
	Vector<Color> to;
	int len = from.size();
	if (len == 0) {
		return Vector<Color>();
	}
	to.resize(len);
	PoolVector<Color>::Read r = from.read();
	Color *w = to.ptrw();
	for (int i = 0; i < len; i++) {
		w[i] = r[i];
	}
	return to;
}

Variant::operator Margin() const {
	return (Margin) operator int();
}
Variant::operator Side() const {
	return (Side) operator int();
}
Variant::operator Orientation() const {
	return (Orientation) operator int();
}

Variant::Variant(bool p_bool) {
	type = BOOL;
	_data._bool = p_bool;
}

/*
Variant::Variant(long unsigned int p_long) {

	type=INT;
	_data._int=p_long;
};
*/

Variant::Variant(signed int p_int) {
	type = INT;
	_data._int = p_int;
}
Variant::Variant(unsigned int p_int) {
	type = INT;
	_data._int = p_int;
}

#ifdef NEED_LONG_INT

Variant::Variant(signed long p_int) {
	type = INT;
	_data._int = p_int;
}
Variant::Variant(unsigned long p_int) {
	type = INT;
	_data._int = p_int;
}
#endif

Variant::Variant(int64_t p_int) {
	type = INT;
	_data._int = p_int;
}

Variant::Variant(uint64_t p_int) {
	type = INT;
	_data._int = p_int;
}

Variant::Variant(signed short p_short) {
	type = INT;
	_data._int = p_short;
}
Variant::Variant(unsigned short p_short) {
	type = INT;
	_data._int = p_short;
}
Variant::Variant(signed char p_char) {
	type = INT;
	_data._int = p_char;
}
Variant::Variant(unsigned char p_char) {
	type = INT;
	_data._int = p_char;
}
Variant::Variant(float p_float) {
	type = REAL;
	_data._real = p_float;
}
Variant::Variant(double p_double) {
	type = REAL;
	_data._real = p_double;
}

Variant::Variant(const StringName &p_string) {
	type = STRING_NAME;
	memnew_placement(_data._mem, StringName(p_string));
}
Variant::Variant(const String &p_string) {
	type = STRING;
	memnew_placement(_data._mem, String(p_string));
}

Variant::Variant(const char *const p_cstring) {
	type = STRING;
	memnew_placement(_data._mem, String((const char *)p_cstring));
}

Variant::Variant(const CharType *p_wstring) {
	type = STRING;
	memnew_placement(_data._mem, String(p_wstring));
}

Variant::Variant(const Rect2 &p_rect2) {
	type = RECT2;
	memnew_placement(_data._mem, Rect2(p_rect2));
}
Variant::Variant(const Rect2i &p_rect2) {
	type = RECT2I;
	memnew_placement(_data._mem, Rect2i(p_rect2));
}

Variant::Variant(const Vector2 &p_vector2) {
	type = VECTOR2;
	memnew_placement(_data._mem, Vector2(p_vector2));
}
Variant::Variant(const Vector2i &p_vector2) {
	type = VECTOR2I;
	memnew_placement(_data._mem, Vector2i(p_vector2));
}

Variant::Variant(const Vector3 &p_vector3) {
	type = VECTOR3;
	memnew_placement(_data._mem, Vector3(p_vector3));
}
Variant::Variant(const Vector3i &p_vector3) {
	type = VECTOR3I;
	memnew_placement(_data._mem, Vector3i(p_vector3));
}

Variant::Variant(const Vector4 &p_vector4) {
	type = VECTOR4;
	memnew_placement(_data._mem, Vector4(p_vector4));
}
Variant::Variant(const Vector4i &p_vector4) {
	type = VECTOR4I;
	memnew_placement(_data._mem, Vector4i(p_vector4));
}

Variant::Variant(const Plane &p_plane) {
	type = PLANE;
	memnew_placement(_data._mem, Plane(p_plane));
}
Variant::Variant(const ::AABB &p_aabb) {
	type = AABB;
	_data._aabb = memnew(::AABB(p_aabb));
}

Variant::Variant(const Basis &p_matrix) {
	type = BASIS;
	_data._basis = memnew(Basis(p_matrix));
}

Variant::Variant(const Quaternion &p_quat) {
	type = QUATERNION;
	memnew_placement(_data._mem, Quaternion(p_quat));
}
Variant::Variant(const Transform &p_transform) {
	type = TRANSFORM;
	_data._transform = memnew(Transform(p_transform));
}

Variant::Variant(const Transform2D &p_transform) {
	type = TRANSFORM2D;
	_data._transform2d = memnew(Transform2D(p_transform));
}

Variant::Variant(const Projection &p_projection) {
	type = PROJECTION;
	_data._projection = memnew(Projection(p_projection));
}

Variant::Variant(const Color &p_color) {
	type = COLOR;
	memnew_placement(_data._mem, Color(p_color));
}

Variant::Variant(const RefPtr &p_resource) {
	type = OBJECT;
	memnew_placement(_data._mem, ObjData);
	_get_obj().rc = nullptr;
	_get_obj().ref = p_resource;
}

Variant::Variant(const Object *p_object) {
	type = OBJECT;
	Object *obj = const_cast<Object *>(p_object);

	memnew_placement(_data._mem, ObjData);
	Reference *ref = Object::cast_to<Reference>(obj);
	if (unlikely(ref)) {
		*reinterpret_cast<Ref<Reference> *>(_get_obj().ref.get_data()) = Ref<Reference>(ref);
		_get_obj().rc = nullptr;
	} else {
		_get_obj().rc = likely(obj) ? obj->_use_rc() : nullptr;
	}
}

Variant::Variant(const Dictionary &p_dictionary) {
	type = DICTIONARY;
	memnew_placement(_data._mem, Dictionary(p_dictionary));
}

Variant::Variant(const Array &p_array) {
	type = ARRAY;
	memnew_placement(_data._mem, Array(p_array));
}

Variant::Variant(const PoolVector<Plane> &p_array) {
	type = ARRAY;

	Array *plane_array = memnew_placement(_data._mem, Array);

	plane_array->resize(p_array.size());

	for (int i = 0; i < p_array.size(); i++) {
		plane_array->operator[](i) = Variant(p_array[i]);
	}
}

Variant::Variant(const Vector<Plane> &p_array) {
	type = ARRAY;

	Array *plane_array = memnew_placement(_data._mem, Array);

	plane_array->resize(p_array.size());

	for (int i = 0; i < p_array.size(); i++) {
		plane_array->operator[](i) = Variant(p_array[i]);
	}
}

Variant::Variant(const PoolVector<uint8_t> &p_raw_array) {
	type = POOL_BYTE_ARRAY;
	memnew_placement(_data._mem, PoolVector<uint8_t>(p_raw_array));
}
Variant::Variant(const PoolVector<int> &p_int_array) {
	type = POOL_INT_ARRAY;
	memnew_placement(_data._mem, PoolVector<int>(p_int_array));
}
Variant::Variant(const PoolVector<real_t> &p_real_array) {
	type = POOL_REAL_ARRAY;
	memnew_placement(_data._mem, PoolVector<real_t>(p_real_array));
}
Variant::Variant(const PoolVector<String> &p_string_array) {
	type = POOL_STRING_ARRAY;
	memnew_placement(_data._mem, PoolVector<String>(p_string_array));
}
Variant::Variant(const PoolVector<Vector2> &p_vector2_array) {
	type = POOL_VECTOR2_ARRAY;
	memnew_placement(_data._mem, PoolVector<Vector2>(p_vector2_array));
}
Variant::Variant(const PoolVector<Vector2i> &p_vector2_array) {
	type = POOL_VECTOR2I_ARRAY;
	memnew_placement(_data._mem, PoolVector<Vector2i>(p_vector2_array));
}
Variant::Variant(const PoolVector<Vector3> &p_vector3_array) {
	type = POOL_VECTOR3_ARRAY;
	memnew_placement(_data._mem, PoolVector<Vector3>(p_vector3_array));
}
Variant::Variant(const PoolVector<Vector3i> &p_vector3_array) {
	type = POOL_VECTOR3I_ARRAY;
	memnew_placement(_data._mem, PoolVector<Vector3i>(p_vector3_array));
}

Variant::Variant(const PoolVector<Vector4> &p_vector4_array) {
	type = POOL_VECTOR4_ARRAY;
	memnew_placement(_data._mem, PoolVector<Vector4>(p_vector4_array));
}
Variant::Variant(const PoolVector<Vector4i> &p_vector4_array) {
	type = POOL_VECTOR4I_ARRAY;
	memnew_placement(_data._mem, PoolVector<Vector4i>(p_vector4_array));
}

Variant::Variant(const PoolVector<Color> &p_color_array) {
	type = POOL_COLOR_ARRAY;
	memnew_placement(_data._mem, PoolVector<Color>(p_color_array));
}

Variant::Variant(const PoolVector<Face3> &p_face_array) {
	PoolVector<Vector3> vertices;
	int face_count = p_face_array.size();
	vertices.resize(face_count * 3);

	if (face_count) {
		PoolVector<Face3>::Read r = p_face_array.read();
		PoolVector<Vector3>::Write w = vertices.write();

		for (int i = 0; i < face_count; i++) {
			for (int j = 0; j < 3; j++) {
				w[i * 3 + j] = r[i].vertex[j];
			}
		}
	}

	type = NIL;

	*this = vertices;
}

/* helpers */

Variant::Variant(const Vector<Variant> &p_array) {
	type = NIL;
	Array v;
	int len = p_array.size();
	v.resize(len);
	for (int i = 0; i < len; i++) {
		v.set(i, p_array[i]);
	}
	*this = v;
}

Variant::Variant(const Vector<uint8_t> &p_array) {
	type = NIL;
	PoolVector<uint8_t> v;
	int len = p_array.size();
	v.resize(len);
	for (int i = 0; i < len; i++) {
		v.set(i, p_array[i]);
	}
	*this = v;
}

Variant::Variant(const Vector<int> &p_array) {
	type = NIL;
	PoolVector<int> v;
	int len = p_array.size();
	v.resize(len);
	for (int i = 0; i < len; i++) {
		v.set(i, p_array[i]);
	}
	*this = v;
}

Variant::Variant(const Vector<real_t> &p_array) {
	type = NIL;
	PoolVector<real_t> v;
	int len = p_array.size();
	v.resize(len);
	for (int i = 0; i < len; i++) {
		v.set(i, p_array[i]);
	}
	*this = v;
}

Variant::Variant(const Vector<String> &p_array) {
	type = NIL;
	PoolVector<String> v;
	int len = p_array.size();
	v.resize(len);
	for (int i = 0; i < len; i++) {
		v.set(i, p_array[i]);
	}
	*this = v;
}

Variant::Variant(const Vector<StringName> &p_array) {
	type = NIL;
	PoolVector<String> v;
	int len = p_array.size();
	v.resize(len);
	for (int i = 0; i < len; i++) {
		v.set(i, p_array[i]);
	}
	*this = v;
}

Variant::Variant(const Vector<Vector2> &p_array) {
	type = NIL;
	PoolVector<Vector2> v;
	int len = p_array.size();
	if (len > 0) {
		v.resize(len);
		PoolVector<Vector2>::Write w = v.write();
		const Vector2 *r = p_array.ptr();

		for (int i = 0; i < len; i++) {
			w[i] = r[i];
		}
	}
	*this = v;
}
Variant::Variant(const Vector<Vector2i> &p_array) {
	type = NIL;
	PoolVector<Vector2i> v;
	int len = p_array.size();
	if (len > 0) {
		v.resize(len);
		PoolVector<Vector2i>::Write w = v.write();
		const Vector2i *r = p_array.ptr();

		for (int i = 0; i < len; i++) {
			w[i] = r[i];
		}
	}
	*this = v;
}

Variant::Variant(const Vector<Vector3> &p_array) {
	type = NIL;
	PoolVector<Vector3> v;
	int len = p_array.size();
	if (len > 0) {
		v.resize(len);
		PoolVector<Vector3>::Write w = v.write();
		const Vector3 *r = p_array.ptr();

		for (int i = 0; i < len; i++) {
			w[i] = r[i];
		}
	}
	*this = v;
}
Variant::Variant(const Vector<Vector3i> &p_array) {
	type = NIL;
	PoolVector<Vector3i> v;
	int len = p_array.size();
	if (len > 0) {
		v.resize(len);
		PoolVector<Vector3i>::Write w = v.write();
		const Vector3i *r = p_array.ptr();

		for (int i = 0; i < len; i++) {
			w[i] = r[i];
		}
	}
	*this = v;
}

Variant::Variant(const Vector<Vector4> &p_array) {
	type = NIL;
	PoolVector<Vector4> v;
	int len = p_array.size();
	if (len > 0) {
		v.resize(len);
		PoolVector<Vector4>::Write w = v.write();
		const Vector4 *r = p_array.ptr();

		for (int i = 0; i < len; i++) {
			w[i] = r[i];
		}
	}
	*this = v;
}
Variant::Variant(const Vector<Vector4i> &p_array) {
	type = NIL;
	PoolVector<Vector4i> v;
	int len = p_array.size();
	if (len > 0) {
		v.resize(len);
		PoolVector<Vector4i>::Write w = v.write();
		const Vector4i *r = p_array.ptr();

		for (int i = 0; i < len; i++) {
			w[i] = r[i];
		}
	}
	*this = v;
}

Variant::Variant(const Vector<Color> &p_array) {
	type = NIL;
	PoolVector<Color> v;
	int len = p_array.size();
	v.resize(len);
	for (int i = 0; i < len; i++) {
		v.set(i, p_array[i]);
	}
	*this = v;
}

void Variant::operator=(const Variant &p_variant) {
	if (unlikely(this == &p_variant)) {
		return;
	}

	if (unlikely(type != p_variant.type)) {
		reference(p_variant);
		return;
	}

	switch (p_variant.type) {
		case NIL: {
			// none
		} break;

		// atomic types
		case BOOL: {
			_data._bool = p_variant._data._bool;
		} break;
		case INT: {
			_data._int = p_variant._data._int;
		} break;
		case REAL: {
			_data._real = p_variant._data._real;
		} break;
		case STRING: {
			*reinterpret_cast<String *>(_data._mem) = *reinterpret_cast<const String *>(p_variant._data._mem);
		} break;

			// math types
		case RECT2: {
			*reinterpret_cast<Rect2 *>(_data._mem) = *reinterpret_cast<const Rect2 *>(p_variant._data._mem);
		} break;
		case RECT2I: {
			*reinterpret_cast<Rect2i *>(_data._mem) = *reinterpret_cast<const Rect2i *>(p_variant._data._mem);
		} break;
		case VECTOR2: {
			*reinterpret_cast<Vector2 *>(_data._mem) = *reinterpret_cast<const Vector2 *>(p_variant._data._mem);
		} break;
		case VECTOR2I: {
			*reinterpret_cast<Vector2i *>(_data._mem) = *reinterpret_cast<const Vector2i *>(p_variant._data._mem);
		} break;
		case VECTOR3: {
			*reinterpret_cast<Vector3 *>(_data._mem) = *reinterpret_cast<const Vector3 *>(p_variant._data._mem);
		} break;
		case VECTOR3I: {
			*reinterpret_cast<Vector3i *>(_data._mem) = *reinterpret_cast<const Vector3i *>(p_variant._data._mem);
		} break;
		case VECTOR4: {
			*reinterpret_cast<Vector4 *>(_data._mem) = *reinterpret_cast<const Vector4 *>(p_variant._data._mem);
		} break;
		case VECTOR4I: {
			*reinterpret_cast<Vector4i *>(_data._mem) = *reinterpret_cast<const Vector4i *>(p_variant._data._mem);
		} break;

		case PLANE: {
			*reinterpret_cast<Plane *>(_data._mem) = *reinterpret_cast<const Plane *>(p_variant._data._mem);
		} break;
		case QUATERNION: {
			*reinterpret_cast<Quaternion *>(_data._mem) = *reinterpret_cast<const Quaternion *>(p_variant._data._mem);
		} break;
		case AABB: {
			*_data._aabb = *(p_variant._data._aabb);
		} break;
		case BASIS: {
			*_data._basis = *(p_variant._data._basis);
		} break;
		case TRANSFORM: {
			*_data._transform = *(p_variant._data._transform);
		} break;
		case TRANSFORM2D: {
			*_data._transform2d = *(p_variant._data._transform2d);
		} break;
		case PROJECTION: {
			*_data._projection = *(p_variant._data._projection);
		} break;

		// misc types
		case COLOR: {
			*reinterpret_cast<Color *>(_data._mem) = *reinterpret_cast<const Color *>(p_variant._data._mem);
		} break;
		case OBJECT: {
			if (likely(_get_obj().rc)) {
				if (unlikely(_get_obj().rc->decrement())) {
					memdelete(_get_obj().rc);
				}
			}
			*reinterpret_cast<ObjData *>(_data._mem) = p_variant._get_obj();
			if (likely(_get_obj().rc)) {
				_get_obj().rc->increment();
			}
		} break;
		case STRING_NAME: {
			*reinterpret_cast<StringName *>(_data._mem) = *reinterpret_cast<const StringName *>(p_variant._data._mem);
		} break;
		case DICTIONARY: {
			*reinterpret_cast<Dictionary *>(_data._mem) = *reinterpret_cast<const Dictionary *>(p_variant._data._mem);
		} break;
		case ARRAY: {
			*reinterpret_cast<Array *>(_data._mem) = *reinterpret_cast<const Array *>(p_variant._data._mem);
		} break;

		// arrays
		case POOL_BYTE_ARRAY: {
			*reinterpret_cast<PoolVector<uint8_t> *>(_data._mem) = *reinterpret_cast<const PoolVector<uint8_t> *>(p_variant._data._mem);
		} break;
		case POOL_INT_ARRAY: {
			*reinterpret_cast<PoolVector<int> *>(_data._mem) = *reinterpret_cast<const PoolVector<int> *>(p_variant._data._mem);
		} break;
		case POOL_REAL_ARRAY: {
			*reinterpret_cast<PoolVector<real_t> *>(_data._mem) = *reinterpret_cast<const PoolVector<real_t> *>(p_variant._data._mem);
		} break;
		case POOL_STRING_ARRAY: {
			*reinterpret_cast<PoolVector<String> *>(_data._mem) = *reinterpret_cast<const PoolVector<String> *>(p_variant._data._mem);
		} break;
		case POOL_VECTOR2_ARRAY: {
			*reinterpret_cast<PoolVector<Vector2> *>(_data._mem) = *reinterpret_cast<const PoolVector<Vector2> *>(p_variant._data._mem);
		} break;
		case POOL_VECTOR2I_ARRAY: {
			*reinterpret_cast<PoolVector<Vector2i> *>(_data._mem) = *reinterpret_cast<const PoolVector<Vector2i> *>(p_variant._data._mem);
		} break;
		case POOL_VECTOR3_ARRAY: {
			*reinterpret_cast<PoolVector<Vector3> *>(_data._mem) = *reinterpret_cast<const PoolVector<Vector3> *>(p_variant._data._mem);
		} break;
		case POOL_VECTOR3I_ARRAY: {
			*reinterpret_cast<PoolVector<Vector3i> *>(_data._mem) = *reinterpret_cast<const PoolVector<Vector3i> *>(p_variant._data._mem);
		} break;
		case POOL_VECTOR4_ARRAY: {
			*reinterpret_cast<PoolVector<Vector4> *>(_data._mem) = *reinterpret_cast<const PoolVector<Vector4> *>(p_variant._data._mem);
		} break;
		case POOL_VECTOR4I_ARRAY: {
			*reinterpret_cast<PoolVector<Vector4i> *>(_data._mem) = *reinterpret_cast<const PoolVector<Vector4i> *>(p_variant._data._mem);
		} break;
		case POOL_COLOR_ARRAY: {
			*reinterpret_cast<PoolVector<Color> *>(_data._mem) = *reinterpret_cast<const PoolVector<Color> *>(p_variant._data._mem);
		} break;
		default: {
		}
	}
}

Variant::Variant(const Variant &p_variant) {
	type = NIL;
	reference(p_variant);
}

/*
Variant::~Variant() {

	clear();
}*/

uint32_t Variant::hash() const {
	return recursive_hash(0);
}

uint32_t Variant::recursive_hash(int p_recursion_count) const {
	switch (type) {
		case NIL: {
			return 0;
		} break;

		case BOOL: {
			return _data._bool ? 1 : 0;
		} break;
		case INT: {
			return hash_one_uint64((uint64_t)_data._int);
		} break;
		case REAL: {
			return hash_murmur3_one_float(_data._real);
		} break;
		case STRING: {
			return reinterpret_cast<const String *>(_data._mem)->hash();
		} break;

			// math types
		case RECT2: {
			return HashMapHasherDefault::hash(*reinterpret_cast<const Rect2 *>(_data._mem));
		} break;
		case RECT2I: {
			return HashMapHasherDefault::hash(*reinterpret_cast<const Rect2i *>(_data._mem));
		} break;
		case VECTOR2: {
			return HashMapHasherDefault::hash(*reinterpret_cast<const Vector2 *>(_data._mem));
		} break;
		case VECTOR2I: {
			return HashMapHasherDefault::hash(*reinterpret_cast<const Vector2i *>(_data._mem));
		} break;
		case VECTOR3: {
			return HashMapHasherDefault::hash(*reinterpret_cast<const Vector3 *>(_data._mem));
		} break;
		case VECTOR3I: {
			return HashMapHasherDefault::hash(*reinterpret_cast<const Vector3i *>(_data._mem));
		} break;
		case VECTOR4: {
			return HashMapHasherDefault::hash(*reinterpret_cast<const Vector4 *>(_data._mem));
		} break;
		case VECTOR4I: {
			return HashMapHasherDefault::hash(*reinterpret_cast<const Vector4i *>(_data._mem));
		} break;
		case PLANE: {
			uint32_t h = HASH_MURMUR3_SEED;
			const Plane &p = *reinterpret_cast<const Plane *>(_data._mem);
			h = hash_murmur3_one_real(p.normal.x, h);
			h = hash_murmur3_one_real(p.normal.y, h);
			h = hash_murmur3_one_real(p.normal.z, h);
			h = hash_murmur3_one_real(p.d, h);
			return hash_fmix32(h);
		} break;
		case QUATERNION: {
			uint32_t h = HASH_MURMUR3_SEED;
			const Quaternion &q = *reinterpret_cast<const Quaternion *>(_data._mem);
			h = hash_murmur3_one_real(q.x, h);
			h = hash_murmur3_one_real(q.y, h);
			h = hash_murmur3_one_real(q.z, h);
			h = hash_murmur3_one_real(q.w, h);
			return hash_fmix32(h);
		} break;
		case AABB: {
			return HashMapHasherDefault::hash(*_data._aabb);
		} break;
		case BASIS: {
			uint32_t h = HASH_MURMUR3_SEED;
			const Basis &b = *_data._basis;
			h = hash_murmur3_one_real(b[0].x, h);
			h = hash_murmur3_one_real(b[0].y, h);
			h = hash_murmur3_one_real(b[0].z, h);
			h = hash_murmur3_one_real(b[1].x, h);
			h = hash_murmur3_one_real(b[1].y, h);
			h = hash_murmur3_one_real(b[1].z, h);
			h = hash_murmur3_one_real(b[2].x, h);
			h = hash_murmur3_one_real(b[2].y, h);
			h = hash_murmur3_one_real(b[2].z, h);
			return hash_fmix32(h);
		} break;
		case TRANSFORM: {
			uint32_t h = HASH_MURMUR3_SEED;
			const Transform &t = *_data._transform;
			h = hash_murmur3_one_real(t.basis[0].x, h);
			h = hash_murmur3_one_real(t.basis[0].y, h);
			h = hash_murmur3_one_real(t.basis[0].z, h);
			h = hash_murmur3_one_real(t.basis[1].x, h);
			h = hash_murmur3_one_real(t.basis[1].y, h);
			h = hash_murmur3_one_real(t.basis[1].z, h);
			h = hash_murmur3_one_real(t.basis[2].x, h);
			h = hash_murmur3_one_real(t.basis[2].y, h);
			h = hash_murmur3_one_real(t.basis[2].z, h);
			h = hash_murmur3_one_real(t.origin.x, h);
			h = hash_murmur3_one_real(t.origin.y, h);
			h = hash_murmur3_one_real(t.origin.z, h);
			return hash_fmix32(h);
		} break;
		case TRANSFORM2D: {
			uint32_t h = HASH_MURMUR3_SEED;
			const Transform2D &t = *_data._transform2d;
			h = hash_murmur3_one_real(t[0].x, h);
			h = hash_murmur3_one_real(t[0].y, h);
			h = hash_murmur3_one_real(t[1].x, h);
			h = hash_murmur3_one_real(t[1].y, h);
			h = hash_murmur3_one_real(t[2].x, h);
			h = hash_murmur3_one_real(t[2].y, h);

			return hash_fmix32(h);
		} break;
		case PROJECTION: {
			uint32_t h = HASH_MURMUR3_SEED;
			const Projection &t = *_data._projection;
			h = hash_murmur3_one_real(t.matrix[0].x, h);
			h = hash_murmur3_one_real(t.matrix[0].y, h);
			h = hash_murmur3_one_real(t.matrix[0].z, h);
			h = hash_murmur3_one_real(t.matrix[0].w, h);
			h = hash_murmur3_one_real(t.matrix[1].x, h);
			h = hash_murmur3_one_real(t.matrix[1].y, h);
			h = hash_murmur3_one_real(t.matrix[1].z, h);
			h = hash_murmur3_one_real(t.matrix[1].w, h);
			h = hash_murmur3_one_real(t.matrix[2].x, h);
			h = hash_murmur3_one_real(t.matrix[2].y, h);
			h = hash_murmur3_one_real(t.matrix[2].z, h);
			h = hash_murmur3_one_real(t.matrix[2].w, h);
			h = hash_murmur3_one_real(t.matrix[3].x, h);
			h = hash_murmur3_one_real(t.matrix[3].y, h);
			h = hash_murmur3_one_real(t.matrix[3].z, h);
			h = hash_murmur3_one_real(t.matrix[3].w, h);
			return hash_fmix32(h);
		} break;

		// misc types
		case COLOR: {
			uint32_t h = HASH_MURMUR3_SEED;
			const Color &c = *reinterpret_cast<const Color *>(_data._mem);
			h = hash_murmur3_one_float(c.r, h);
			h = hash_murmur3_one_float(c.g, h);
			h = hash_murmur3_one_float(c.b, h);
			h = hash_murmur3_one_float(c.a, h);
			return hash_fmix32(h);
		} break;
		case OBJECT: {
			return hash_one_uint64(hash_make_uint64_t(_UNSAFE_OBJ_PROXY_PTR(*this)));
		} break;
		case STRING_NAME: {
			return reinterpret_cast<const StringName *>(_data._mem)->hash();
		} break;
		case DICTIONARY: {
			return reinterpret_cast<const Dictionary *>(_data._mem)->recursive_hash(p_recursion_count);
		} break;
		case ARRAY: {
			const Array &arr = *reinterpret_cast<const Array *>(_data._mem);
			return arr.recursive_hash(p_recursion_count);
		} break;

		case POOL_BYTE_ARRAY: {
			const PoolVector<uint8_t> &arr = *reinterpret_cast<const PoolVector<uint8_t> *>(_data._mem);
			int len = arr.size();
			if (likely(len)) {
				PoolVector<uint8_t>::Read r = arr.read();
				return hash_murmur3_buffer((uint8_t *)&r[0], len);
			} else {
				return hash_murmur3_one_64(0);
			}

		} break;
		case POOL_INT_ARRAY: {
			const PoolVector<int> &arr = *reinterpret_cast<const PoolVector<int> *>(_data._mem);
			int len = arr.size();
			if (likely(len)) {
				PoolVector<int>::Read r = arr.read();
				return hash_murmur3_buffer((uint8_t *)&r[0], len * sizeof(int));
			} else {
				return hash_murmur3_one_64(0);
			}

		} break;
		case POOL_REAL_ARRAY: {
			const PoolVector<real_t> &arr = *reinterpret_cast<const PoolVector<real_t> *>(_data._mem);
			int len = arr.size();

			if (likely(len)) {
				PoolVector<real_t>::Read r = arr.read();
				uint32_t h = HASH_MURMUR3_SEED;

				for (int i = 0; i < len; i++) {
					h = hash_murmur3_one_real(r[i], h);
				}

				return hash_fmix32(h);
			} else {
				return hash_murmur3_one_real(0.0);
			}

		} break;
		case POOL_STRING_ARRAY: {
			uint32_t hash = HASH_MURMUR3_SEED;
			const PoolVector<String> &arr = *reinterpret_cast<const PoolVector<String> *>(_data._mem);
			int len = arr.size();

			if (likely(len)) {
				PoolVector<String>::Read r = arr.read();

				for (int i = 0; i < len; i++) {
					hash = hash_murmur3_one_32(r[i].hash(), hash);
				}

				hash = hash_fmix32(hash);
			}

			return hash;
		} break;
		case POOL_VECTOR2_ARRAY: {
			uint32_t hash = HASH_MURMUR3_SEED;
			const PoolVector<Vector2> &arr = *reinterpret_cast<const PoolVector<Vector2> *>(_data._mem);
			int len = arr.size();

			if (likely(len)) {
				PoolVector<Vector2>::Read r = arr.read();

				for (int i = 0; i < len; i++) {
					hash = hash_murmur3_one_real(r[i].x, hash);
					hash = hash_murmur3_one_real(r[i].y, hash);
				}

				hash = hash_fmix32(hash);
			}

			return hash;
		} break;
		case POOL_VECTOR2I_ARRAY: {
			uint32_t hash = HASH_MURMUR3_SEED;
			const PoolVector<Vector2i> &arr = *reinterpret_cast<const PoolVector<Vector2i> *>(_data._mem);
			int len = arr.size();

			if (likely(len)) {
				PoolVector<Vector2i>::Read r = arr.read();

				for (int i = 0; i < len; i++) {
					hash = hash_murmur3_one_32(r[i].x, hash);
					hash = hash_murmur3_one_32(r[i].y, hash);
				}

				hash = hash_fmix32(hash);
			}

			return hash;
		} break;
		case POOL_VECTOR3_ARRAY: {
			uint32_t hash = HASH_MURMUR3_SEED;
			const PoolVector<Vector3> &arr = *reinterpret_cast<const PoolVector<Vector3> *>(_data._mem);
			int len = arr.size();

			if (likely(len)) {
				PoolVector<Vector3>::Read r = arr.read();

				for (int i = 0; i < len; i++) {
					hash = hash_murmur3_one_real(r[i].x, hash);
					hash = hash_murmur3_one_real(r[i].y, hash);
					hash = hash_murmur3_one_real(r[i].z, hash);
				}

				hash = hash_fmix32(hash);
			}

			return hash;
		} break;
		case POOL_VECTOR3I_ARRAY: {
			uint32_t hash = HASH_MURMUR3_SEED;
			const PoolVector<Vector3i> &arr = *reinterpret_cast<const PoolVector<Vector3i> *>(_data._mem);
			int len = arr.size();

			if (likely(len)) {
				PoolVector<Vector3i>::Read r = arr.read();

				for (int i = 0; i < len; i++) {
					hash = hash_murmur3_one_32(r[i].x, hash);
					hash = hash_murmur3_one_32(r[i].y, hash);
					hash = hash_murmur3_one_32(r[i].z, hash);
				}

				hash = hash_fmix32(hash);
			}

			return hash;
		} break;
		case POOL_VECTOR4_ARRAY: {
			uint32_t hash = HASH_MURMUR3_SEED;
			const PoolVector<Vector4> &arr = *reinterpret_cast<const PoolVector<Vector4> *>(_data._mem);
			int len = arr.size();

			if (likely(len)) {
				PoolVector<Vector4>::Read r = arr.read();

				for (int i = 0; i < len; i++) {
					hash = hash_murmur3_one_real(r[i].x, hash);
					hash = hash_murmur3_one_real(r[i].y, hash);
					hash = hash_murmur3_one_real(r[i].z, hash);
					hash = hash_murmur3_one_real(r[i].w, hash);
				}

				hash = hash_fmix32(hash);
			}

			return hash;
		} break;
		case POOL_VECTOR4I_ARRAY: {
			uint32_t hash = HASH_MURMUR3_SEED;
			const PoolVector<Vector4i> &arr = *reinterpret_cast<const PoolVector<Vector4i> *>(_data._mem);
			int len = arr.size();

			if (likely(len)) {
				PoolVector<Vector4i>::Read r = arr.read();

				for (int i = 0; i < len; i++) {
					hash = hash_murmur3_one_32(r[i].x, hash);
					hash = hash_murmur3_one_32(r[i].y, hash);
					hash = hash_murmur3_one_32(r[i].z, hash);
					hash = hash_murmur3_one_32(r[i].w, hash);
				}

				hash = hash_fmix32(hash);
			}

			return hash;
		} break;
		case POOL_COLOR_ARRAY: {
			uint32_t hash = HASH_MURMUR3_SEED;
			const PoolVector<Color> &arr = *reinterpret_cast<const PoolVector<Color> *>(_data._mem);
			int len = arr.size();

			if (likely(len)) {
				PoolVector<Color>::Read r = arr.read();

				for (int i = 0; i < len; i++) {
					hash = hash_murmur3_one_real(r[i].r, hash);
					hash = hash_murmur3_one_real(r[i].g, hash);
					hash = hash_murmur3_one_real(r[i].b, hash);
					hash = hash_murmur3_one_real(r[i].a, hash);
				}

				hash = hash_fmix32(hash);
			}

			return hash;
		} break;
		default: {
		}
	}

	return 0;
}

#define hash_compare_scalar(p_lhs, p_rhs) \
	(((p_lhs) == (p_rhs)) || (Math::is_nan(p_lhs) && Math::is_nan(p_rhs)))

#define hash_compare_vector2(p_lhs, p_rhs)        \
	(hash_compare_scalar((p_lhs).x, (p_rhs).x) && \
			hash_compare_scalar((p_lhs).y, (p_rhs).y))

#define hash_compare_vector2i(p_lhs, p_rhs) \
	(((p_lhs).x == (p_rhs).x) &&            \
			((p_lhs).y == (p_rhs).y))

#define hash_compare_vector3(p_lhs, p_rhs)               \
	(hash_compare_scalar((p_lhs).x, (p_rhs).x) &&        \
			hash_compare_scalar((p_lhs).y, (p_rhs).y) && \
			hash_compare_scalar((p_lhs).z, (p_rhs).z))

#define hash_compare_vector3i(p_lhs, p_rhs) \
	(((p_lhs).x == (p_rhs).x) &&            \
			((p_lhs).y == (p_rhs).y) &&     \
			((p_lhs).z == (p_rhs).z))

#define hash_compare_vector4(p_lhs, p_rhs)               \
	(hash_compare_scalar((p_lhs).x, (p_rhs).x) &&        \
			hash_compare_scalar((p_lhs).y, (p_rhs).y) && \
			hash_compare_scalar((p_lhs).z, (p_rhs).z) && \
			hash_compare_scalar((p_lhs).w, (p_rhs).w))

#define hash_compare_vector4i(p_lhs, p_rhs) \
	(((p_lhs).x == (p_rhs).x) &&            \
			((p_lhs).y == (p_rhs).y) &&     \
			((p_lhs).z == (p_rhs).z) &&     \
			((p_lhs).w == (p_rhs).w))

#define hash_compare_quat(p_lhs, p_rhs)                  \
	(hash_compare_scalar((p_lhs).x, (p_rhs).x) &&        \
			hash_compare_scalar((p_lhs).y, (p_rhs).y) && \
			hash_compare_scalar((p_lhs).z, (p_rhs).z) && \
			hash_compare_scalar((p_lhs).w, (p_rhs).w))

#define hash_compare_color(p_lhs, p_rhs)                 \
	(hash_compare_scalar((p_lhs).r, (p_rhs).r) &&        \
			hash_compare_scalar((p_lhs).g, (p_rhs).g) && \
			hash_compare_scalar((p_lhs).b, (p_rhs).b) && \
			hash_compare_scalar((p_lhs).a, (p_rhs).a))

#define hash_compare_pool_array(p_lhs, p_rhs, p_type, p_compare_func)                   \
	const PoolVector<p_type> &l = *reinterpret_cast<const PoolVector<p_type> *>(p_lhs); \
	const PoolVector<p_type> &r = *reinterpret_cast<const PoolVector<p_type> *>(p_rhs); \
                                                                                        \
	if (l.size() != r.size())                                                           \
		return false;                                                                   \
                                                                                        \
	PoolVector<p_type>::Read lr = l.read();                                             \
	PoolVector<p_type>::Read rr = r.read();                                             \
                                                                                        \
	for (int i = 0; i < l.size(); ++i) {                                                \
		if (!p_compare_func((lr[i]), (rr[i])))                                          \
			return false;                                                               \
	}                                                                                   \
                                                                                        \
	return true

bool Variant::hash_compare(const Variant &p_variant) const {
	if (type != p_variant.type) {
		return false;
	}

	switch (type) {
		//BOOL
		case INT: {
			return _data._int == p_variant._data._int;
		} break;
		case REAL: {
			return hash_compare_scalar(_data._real, p_variant._data._real);
		} break;
		case STRING: {
			return *reinterpret_cast<const String *>(_data._mem) == *reinterpret_cast<const String *>(p_variant._data._mem);
		} break;

		case RECT2: {
			const Rect2 *l = reinterpret_cast<const Rect2 *>(_data._mem);
			const Rect2 *r = reinterpret_cast<const Rect2 *>(p_variant._data._mem);

			return hash_compare_vector2(l->position, r->position) &&
					hash_compare_vector2(l->size, r->size);
		} break;
		case RECT2I: {
			const Rect2i *l = reinterpret_cast<const Rect2i *>(_data._mem);
			const Rect2i *r = reinterpret_cast<const Rect2i *>(p_variant._data._mem);

			return hash_compare_vector2i(l->position, r->position) &&
					hash_compare_vector2i(l->size, r->size);
		} break;
		case VECTOR2: {
			const Vector2 *l = reinterpret_cast<const Vector2 *>(_data._mem);
			const Vector2 *r = reinterpret_cast<const Vector2 *>(p_variant._data._mem);

			return hash_compare_vector2(*l, *r);
		} break;
		case VECTOR2I: {
			const Vector2i *l = reinterpret_cast<const Vector2i *>(_data._mem);
			const Vector2i *r = reinterpret_cast<const Vector2i *>(p_variant._data._mem);

			return hash_compare_vector2i(*l, *r);
		} break;
		case VECTOR3: {
			const Vector3 *l = reinterpret_cast<const Vector3 *>(_data._mem);
			const Vector3 *r = reinterpret_cast<const Vector3 *>(p_variant._data._mem);

			return hash_compare_vector3(*l, *r);
		} break;
		case VECTOR3I: {
			const Vector3i *l = reinterpret_cast<const Vector3i *>(_data._mem);
			const Vector3i *r = reinterpret_cast<const Vector3i *>(p_variant._data._mem);

			return hash_compare_vector3i(*l, *r);
		} break;
		case VECTOR4: {
			const Vector4 *l = reinterpret_cast<const Vector4 *>(_data._mem);
			const Vector4 *r = reinterpret_cast<const Vector4 *>(p_variant._data._mem);

			return hash_compare_vector4(*l, *r);
		} break;
		case VECTOR4I: {
			const Vector4i *l = reinterpret_cast<const Vector4i *>(_data._mem);
			const Vector4i *r = reinterpret_cast<const Vector4i *>(p_variant._data._mem);

			return hash_compare_vector4i(*l, *r);
		} break;

		case PLANE: {
			const Plane *l = reinterpret_cast<const Plane *>(_data._mem);
			const Plane *r = reinterpret_cast<const Plane *>(p_variant._data._mem);

			return hash_compare_vector3(l->normal, r->normal) &&
					hash_compare_scalar(l->d, r->d);
		} break;
		case QUATERNION: {
			const Quaternion *l = reinterpret_cast<const Quaternion *>(_data._mem);
			const Quaternion *r = reinterpret_cast<const Quaternion *>(p_variant._data._mem);

			return hash_compare_quat(*l, *r);
		} break;
		case AABB: {
			const ::AABB *l = _data._aabb;
			const ::AABB *r = p_variant._data._aabb;

			return hash_compare_vector3(l->position, r->position) &&
					hash_compare_vector3(l->size, r->size);

		} break;
		case BASIS: {
			const Basis *l = _data._basis;
			const Basis *r = p_variant._data._basis;

			for (int i = 0; i < 3; i++) {
				if (!hash_compare_vector3(l->rows[i], r->rows[i])) {
					return false;
				}
			}

			return true;
		} break;
		case TRANSFORM: {
			const Transform *l = _data._transform;
			const Transform *r = p_variant._data._transform;

			for (int i = 0; i < 3; i++) {
				if (!hash_compare_vector3(l->basis.rows[i], r->basis.rows[i])) {
					return false;
				}
			}

			return hash_compare_vector3(l->origin, r->origin);
		} break;
		case TRANSFORM2D: {
			Transform2D *l = _data._transform2d;
			Transform2D *r = p_variant._data._transform2d;

			for (int i = 0; i < 3; i++) {
				if (!hash_compare_vector2(l->columns[i], r->columns[i])) {
					return false;
				}
			}

			return true;
		} break;
		case PROJECTION: {
			const Projection *l = _data._projection;
			const Projection *r = p_variant._data._projection;

			for (int i = 0; i < 4; i++) {
				if (!hash_compare_vector4(l->matrix[i], r->matrix[i])) {
					return false;
				}
			}

			return true;
		} break;

		case COLOR: {
			const Color *l = reinterpret_cast<const Color *>(_data._mem);
			const Color *r = reinterpret_cast<const Color *>(p_variant._data._mem);

			return hash_compare_color(*l, *r);
		} break;
		case ARRAY: {
			const Array &l = *(reinterpret_cast<const Array *>(_data._mem));
			const Array &r = *(reinterpret_cast<const Array *>(p_variant._data._mem));

			if (l.size() != r.size()) {
				return false;
			}

			for (int i = 0; i < l.size(); ++i) {
				if (!l[i].hash_compare(r[i])) {
					return false;
				}
			}

			return true;
		} break;

		case POOL_REAL_ARRAY: {
			hash_compare_pool_array(_data._mem, p_variant._data._mem, real_t, hash_compare_scalar);
		} break;
		case POOL_VECTOR2_ARRAY: {
			hash_compare_pool_array(_data._mem, p_variant._data._mem, Vector2, hash_compare_vector2);
		} break;
		case POOL_VECTOR2I_ARRAY: {
			hash_compare_pool_array(_data._mem, p_variant._data._mem, Vector2i, hash_compare_vector2i);
		} break;
		case POOL_VECTOR3_ARRAY: {
			hash_compare_pool_array(_data._mem, p_variant._data._mem, Vector3, hash_compare_vector3);
		} break;
		case POOL_VECTOR3I_ARRAY: {
			hash_compare_pool_array(_data._mem, p_variant._data._mem, Vector3i, hash_compare_vector3i);
		} break;
		case POOL_VECTOR4_ARRAY: {
			hash_compare_pool_array(_data._mem, p_variant._data._mem, Vector4, hash_compare_vector4);
		} break;
		case POOL_VECTOR4I_ARRAY: {
			hash_compare_pool_array(_data._mem, p_variant._data._mem, Vector4i, hash_compare_vector4i);
		} break;
		case POOL_COLOR_ARRAY: {
			hash_compare_pool_array(_data._mem, p_variant._data._mem, Color, hash_compare_color);
		} break;

		default:
			bool v;
			Variant r;
			evaluate(OP_EQUAL, *this, p_variant, r, v);
			return r;
	}

	return false;
}

bool Variant::is_ref() const {
	return type == OBJECT && !_get_obj().ref.is_null();
}

Vector<Variant> varray() {
	return Vector<Variant>();
}

Vector<Variant> varray(const Variant &p_arg1) {
	Vector<Variant> v;
	v.push_back(p_arg1);
	return v;
}
Vector<Variant> varray(const Variant &p_arg1, const Variant &p_arg2) {
	Vector<Variant> v;
	v.push_back(p_arg1);
	v.push_back(p_arg2);
	return v;
}
Vector<Variant> varray(const Variant &p_arg1, const Variant &p_arg2, const Variant &p_arg3) {
	Vector<Variant> v;
	v.push_back(p_arg1);
	v.push_back(p_arg2);
	v.push_back(p_arg3);
	return v;
}
Vector<Variant> varray(const Variant &p_arg1, const Variant &p_arg2, const Variant &p_arg3, const Variant &p_arg4) {
	Vector<Variant> v;
	v.push_back(p_arg1);
	v.push_back(p_arg2);
	v.push_back(p_arg3);
	v.push_back(p_arg4);
	return v;
}

Vector<Variant> varray(const Variant &p_arg1, const Variant &p_arg2, const Variant &p_arg3, const Variant &p_arg4, const Variant &p_arg5) {
	Vector<Variant> v;
	v.push_back(p_arg1);
	v.push_back(p_arg2);
	v.push_back(p_arg3);
	v.push_back(p_arg4);
	v.push_back(p_arg5);
	return v;
}

bool Variant::is_shared() const {
	switch (type) {
		case OBJECT:
			return true;
		case ARRAY:
			return true;
		case DICTIONARY:
			return true;
		default: {
		}
	}

	return false;
}

String vformat(const String &p_text, const Variant &p1, const Variant &p2, const Variant &p3, const Variant &p4, const Variant &p5) {
	Array args;
	if (p1.get_type() != Variant::NIL) {
		args.push_back(p1);

		if (p2.get_type() != Variant::NIL) {
			args.push_back(p2);

			if (p3.get_type() != Variant::NIL) {
				args.push_back(p3);

				if (p4.get_type() != Variant::NIL) {
					args.push_back(p4);

					if (p5.get_type() != Variant::NIL) {
						args.push_back(p5);
					}
				}
			}
		}
	}

	bool error = false;
	String fmt = args.sprintf(p_text, &error);

	ERR_FAIL_COND_V_MSG(error, String(), fmt);

	return fmt;
}
#line 0

#line 1 "sfw/object/variant_op.cpp"
/*************************************************************************/
/*  variant_op.cpp                                                       */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

#define CASE_TYPE_ALL(PREFIX, OP) \
	CASE_TYPE(PREFIX, OP, INT)    \
	CASE_TYPE_ALL_BUT_INT(PREFIX, OP)

#define CASE_TYPE_ALL_BUT_INT(PREFIX, OP)      \
	CASE_TYPE(PREFIX, OP, NIL)                 \
	CASE_TYPE(PREFIX, OP, BOOL)                \
	CASE_TYPE(PREFIX, OP, REAL)                \
	CASE_TYPE(PREFIX, OP, STRING)              \
	CASE_TYPE(PREFIX, OP, RECT2)               \
	CASE_TYPE(PREFIX, OP, RECT2I)              \
	CASE_TYPE(PREFIX, OP, VECTOR2)             \
	CASE_TYPE(PREFIX, OP, VECTOR2I)            \
	CASE_TYPE(PREFIX, OP, VECTOR3)             \
	CASE_TYPE(PREFIX, OP, VECTOR3I)            \
	CASE_TYPE(PREFIX, OP, VECTOR4)             \
	CASE_TYPE(PREFIX, OP, VECTOR4I)            \
	CASE_TYPE(PREFIX, OP, PLANE)               \
	CASE_TYPE(PREFIX, OP, QUATERNION)          \
	CASE_TYPE(PREFIX, OP, AABB)                \
	CASE_TYPE(PREFIX, OP, BASIS)               \
	CASE_TYPE(PREFIX, OP, TRANSFORM)           \
	CASE_TYPE(PREFIX, OP, TRANSFORM2D)         \
	CASE_TYPE(PREFIX, OP, PROJECTION)          \
	CASE_TYPE(PREFIX, OP, COLOR)               \
	CASE_TYPE(PREFIX, OP, OBJECT)              \
	CASE_TYPE(PREFIX, OP, STRING_NAME)         \
	CASE_TYPE(PREFIX, OP, DICTIONARY)          \
	CASE_TYPE(PREFIX, OP, ARRAY)               \
	CASE_TYPE(PREFIX, OP, POOL_BYTE_ARRAY)     \
	CASE_TYPE(PREFIX, OP, POOL_INT_ARRAY)      \
	CASE_TYPE(PREFIX, OP, POOL_REAL_ARRAY)     \
	CASE_TYPE(PREFIX, OP, POOL_STRING_ARRAY)   \
	CASE_TYPE(PREFIX, OP, POOL_VECTOR2_ARRAY)  \
	CASE_TYPE(PREFIX, OP, POOL_VECTOR2I_ARRAY) \
	CASE_TYPE(PREFIX, OP, POOL_VECTOR3_ARRAY)  \
	CASE_TYPE(PREFIX, OP, POOL_VECTOR3I_ARRAY) \
	CASE_TYPE(PREFIX, OP, POOL_VECTOR4_ARRAY)  \
	CASE_TYPE(PREFIX, OP, POOL_VECTOR4I_ARRAY) \
	CASE_TYPE(PREFIX, OP, POOL_COLOR_ARRAY)

#ifdef __GNUC__
#define TYPE(PREFIX, OP, TYPE) &&PREFIX##_##OP##_##TYPE

/* clang-format off */

#define TYPES(PREFIX, OP) {                    \
		TYPE(PREFIX, OP, NIL),                 \
		TYPE(PREFIX, OP, BOOL),                \
		TYPE(PREFIX, OP, INT),                 \
		TYPE(PREFIX, OP, REAL),                \
		TYPE(PREFIX, OP, STRING),              \
		TYPE(PREFIX, OP, RECT2),               \
		TYPE(PREFIX, OP, RECT2I),              \
		TYPE(PREFIX, OP, VECTOR2),             \
		TYPE(PREFIX, OP, VECTOR2I),            \
		TYPE(PREFIX, OP, VECTOR3),             \
		TYPE(PREFIX, OP, VECTOR3I),            \
		TYPE(PREFIX, OP, VECTOR4),             \
		TYPE(PREFIX, OP, VECTOR4I),            \
		TYPE(PREFIX, OP, PLANE),               \
		TYPE(PREFIX, OP, QUATERNION),          \
		TYPE(PREFIX, OP, AABB),                \
		TYPE(PREFIX, OP, BASIS),               \
		TYPE(PREFIX, OP, TRANSFORM),           \
		TYPE(PREFIX, OP, TRANSFORM2D),         \
		TYPE(PREFIX, OP, PROJECTION),          \
		TYPE(PREFIX, OP, COLOR),               \
		TYPE(PREFIX, OP, OBJECT),              \
		TYPE(PREFIX, OP, STRING_NAME),         \
		TYPE(PREFIX, OP, DICTIONARY),          \
		TYPE(PREFIX, OP, ARRAY),               \
		TYPE(PREFIX, OP, POOL_BYTE_ARRAY),     \
		TYPE(PREFIX, OP, POOL_INT_ARRAY),      \
		TYPE(PREFIX, OP, POOL_REAL_ARRAY),     \
		TYPE(PREFIX, OP, POOL_STRING_ARRAY),   \
		TYPE(PREFIX, OP, POOL_VECTOR2_ARRAY),  \
		TYPE(PREFIX, OP, POOL_VECTOR2I_ARRAY), \
		TYPE(PREFIX, OP, POOL_VECTOR3_ARRAY),  \
		TYPE(PREFIX, OP, POOL_VECTOR3I_ARRAY), \
		TYPE(PREFIX, OP, POOL_VECTOR4_ARRAY),  \
		TYPE(PREFIX, OP, POOL_VECTOR4I_ARRAY), \
		TYPE(PREFIX, OP, POOL_COLOR_ARRAY),    \
}

/* clang-format on */

#define CASES(PREFIX) static const void *switch_table_##PREFIX[25][38] = { \
	TYPES(PREFIX, OP_EQUAL),                                               \
	TYPES(PREFIX, OP_NOT_EQUAL),                                           \
	TYPES(PREFIX, OP_LESS),                                                \
	TYPES(PREFIX, OP_LESS_EQUAL),                                          \
	TYPES(PREFIX, OP_GREATER),                                             \
	TYPES(PREFIX, OP_GREATER_EQUAL),                                       \
	TYPES(PREFIX, OP_ADD),                                                 \
	TYPES(PREFIX, OP_SUBTRACT),                                            \
	TYPES(PREFIX, OP_MULTIPLY),                                            \
	TYPES(PREFIX, OP_DIVIDE),                                              \
	TYPES(PREFIX, OP_NEGATE),                                              \
	TYPES(PREFIX, OP_POSITIVE),                                            \
	TYPES(PREFIX, OP_MODULE),                                              \
	TYPES(PREFIX, OP_STRING_CONCAT),                                       \
	TYPES(PREFIX, OP_SHIFT_LEFT),                                          \
	TYPES(PREFIX, OP_SHIFT_RIGHT),                                         \
	TYPES(PREFIX, OP_BIT_AND),                                             \
	TYPES(PREFIX, OP_BIT_OR),                                              \
	TYPES(PREFIX, OP_BIT_XOR),                                             \
	TYPES(PREFIX, OP_BIT_NEGATE),                                          \
	TYPES(PREFIX, OP_AND),                                                 \
	TYPES(PREFIX, OP_OR),                                                  \
	TYPES(PREFIX, OP_XOR),                                                 \
	TYPES(PREFIX, OP_NOT),                                                 \
	TYPES(PREFIX, OP_IN),                                                  \
}

#define SWITCH(PREFIX, op, val) goto *switch_table_##PREFIX[op][val];
#define SWITCH_OP(PREFIX, OP, val)
#define CASE_TYPE(PREFIX, OP, TYPE) PREFIX##_##OP##_##TYPE:

#else
#define CASES(PREFIX)
#define SWITCH(PREFIX, op, val) switch (op)
#define SWITCH_OP(PREFIX, OP, val) \
	case OP:                       \
		switch (val)
#define CASE_TYPE(PREFIX, OP, TYPE) case TYPE:
#endif

Variant::operator bool() const {
	return booleanize();
}

// We consider all uninitialized or empty types to be false based on the type's
// zeroiness.
bool Variant::booleanize() const {
	return !is_zero();
}

#define _RETURN(m_what) \
	{                   \
		r_ret = m_what; \
		return;         \
	}

#define _RETURN_FAIL     \
	{                    \
		r_valid = false; \
		return;          \
	}

#define DEFAULT_OP_NUM(m_prefix, m_op_name, m_name, m_op, m_type) \
	CASE_TYPE(m_prefix, m_op_name, m_name) {                      \
		if (p_b.type == INT)                                      \
			_RETURN(p_a._data.m_type m_op p_b._data._int);        \
		if (p_b.type == REAL)                                     \
			_RETURN(p_a._data.m_type m_op p_b._data._real);       \
                                                                  \
		_RETURN_FAIL                                              \
	};

#define DEFAULT_OP_NUM_NULL(m_prefix, m_op_name, m_name, m_op, m_type) \
	CASE_TYPE(m_prefix, m_op_name, m_name) {                           \
		if (p_b.type == INT)                                           \
			_RETURN(p_a._data.m_type m_op p_b._data._int);             \
		if (p_b.type == REAL)                                          \
			_RETURN(p_a._data.m_type m_op p_b._data._real);            \
		if (p_b.type == NIL)                                           \
			_RETURN(!(p_b.type m_op NIL));                             \
                                                                       \
		_RETURN_FAIL                                                   \
	};

#ifdef DEBUG_ENABLED
#define DEFAULT_OP_NUM_DIV(m_prefix, m_op_name, m_name, m_type) \
	CASE_TYPE(m_prefix, m_op_name, m_name) {                    \
		if (p_b.type == INT) {                                  \
			if (p_b._data._int == 0) {                          \
				r_valid = false;                                \
				_RETURN("Division By Zero");                    \
			}                                                   \
			_RETURN(p_a._data.m_type / p_b._data._int);         \
		}                                                       \
		if (p_b.type == REAL) {                                 \
			if (p_b._data._real == 0) {                         \
				r_valid = false;                                \
				_RETURN("Division By Zero");                    \
			}                                                   \
			_RETURN(p_a._data.m_type / p_b._data._real);        \
		}                                                       \
                                                                \
		_RETURN_FAIL                                            \
	};
#else
#define DEFAULT_OP_NUM_DIV(m_prefix, m_op_name, m_name, m_type) \
	CASE_TYPE(m_prefix, m_op_name, m_name) {                    \
		if (p_b.type == INT)                                    \
			_RETURN(p_a._data.m_type / p_b._data._int);         \
		if (p_b.type == REAL)                                   \
			_RETURN(p_a._data.m_type / p_b._data._real);        \
                                                                \
		_RETURN_FAIL                                            \
	};
#endif

#define DEFAULT_OP_NUM_NEG(m_prefix, m_op_name, m_name, m_type) \
	CASE_TYPE(m_prefix, m_op_name, m_name) {                    \
		_RETURN(-p_a._data.m_type);                             \
	};

#define DEFAULT_OP_NUM_POS(m_prefix, m_op_name, m_name, m_type) \
	CASE_TYPE(m_prefix, m_op_name, m_name) {                    \
		_RETURN(p_a._data.m_type);                              \
	};

#define DEFAULT_OP_NUM_VEC(m_prefix, m_op_name, m_name, m_op, m_type)                           \
	CASE_TYPE(m_prefix, m_op_name, m_name) {                                                    \
		if (p_b.type == INT)                                                                    \
			_RETURN(p_a._data.m_type m_op p_b._data._int);                                      \
		if (p_b.type == REAL)                                                                   \
			_RETURN(p_a._data.m_type m_op p_b._data._real);                                     \
		if (p_b.type == VECTOR2)                                                                \
			_RETURN(p_a._data.m_type m_op *reinterpret_cast<const Vector2 *>(p_b._data._mem));  \
		if (p_b.type == VECTOR2I)                                                               \
			_RETURN(p_a._data.m_type m_op *reinterpret_cast<const Vector2i *>(p_b._data._mem)); \
		if (p_b.type == VECTOR3)                                                                \
			_RETURN(p_a._data.m_type m_op *reinterpret_cast<const Vector3 *>(p_b._data._mem));  \
		if (p_b.type == VECTOR3I)                                                               \
			_RETURN(p_a._data.m_type m_op *reinterpret_cast<const Vector3i *>(p_b._data._mem)); \
		if (p_b.type == VECTOR4)                                                                \
			_RETURN(p_a._data.m_type m_op *reinterpret_cast<const Vector4 *>(p_b._data._mem));  \
		if (p_b.type == VECTOR4I)                                                               \
			_RETURN(p_a._data.m_type m_op *reinterpret_cast<const Vector4i *>(p_b._data._mem)); \
                                                                                                \
		_RETURN_FAIL                                                                            \
	};

#define DEFAULT_OP_STR_REV(m_prefix, m_op_name, m_name, m_op, m_type)                                                              \
	CASE_TYPE(m_prefix, m_op_name, m_name) {                                                                                       \
		if (p_b.type == STRING)                                                                                                    \
			_RETURN(*reinterpret_cast<const m_type *>(p_b._data._mem) m_op *reinterpret_cast<const String *>(p_a._data._mem));     \
		if (p_b.type == STRING_NAME)                                                                                               \
			_RETURN(*reinterpret_cast<const m_type *>(p_b._data._mem) m_op *reinterpret_cast<const StringName *>(p_a._data._mem)); \
                                                                                                                                   \
		_RETURN_FAIL                                                                                                               \
	};

#define DEFAULT_OP_STR(m_prefix, m_op_name, m_name, m_op, m_type)                                                                  \
	CASE_TYPE(m_prefix, m_op_name, m_name) {                                                                                       \
		if (p_b.type == STRING)                                                                                                    \
			_RETURN(*reinterpret_cast<const m_type *>(p_a._data._mem) m_op *reinterpret_cast<const String *>(p_b._data._mem));     \
		if (p_b.type == STRING_NAME)                                                                                               \
			_RETURN(*reinterpret_cast<const m_type *>(p_a._data._mem) m_op *reinterpret_cast<const StringName *>(p_b._data._mem)); \
                                                                                                                                   \
		_RETURN_FAIL                                                                                                               \
	};

#define DEFAULT_OP_STR_NULL(m_prefix, m_op_name, m_name, m_op, m_type)                                                             \
	CASE_TYPE(m_prefix, m_op_name, m_name) {                                                                                       \
		if (p_b.type == STRING)                                                                                                    \
			_RETURN(*reinterpret_cast<const m_type *>(p_a._data._mem) m_op *reinterpret_cast<const String *>(p_b._data._mem));     \
		if (p_b.type == STRING_NAME)                                                                                               \
			_RETURN(*reinterpret_cast<const m_type *>(p_a._data._mem) m_op *reinterpret_cast<const StringName *>(p_b._data._mem)); \
		if (p_b.type == NIL)                                                                                                       \
			_RETURN(!(p_b.type m_op NIL));                                                                                         \
                                                                                                                                   \
		_RETURN_FAIL                                                                                                               \
	};

#define DEFAULT_OP_STR_NULL_NP(m_prefix, m_op_name, m_name, m_op, m_type)                                                      \
	CASE_TYPE(m_prefix, m_op_name, m_name) {                                                                                   \
		if (p_b.type == STRING)                                                                                                \
			_RETURN(*reinterpret_cast<const m_type *>(p_a._data._mem) m_op *reinterpret_cast<const String *>(p_b._data._mem)); \
		if (p_b.type == NIL)                                                                                                   \
			_RETURN(!(p_b.type m_op NIL));                                                                                     \
                                                                                                                               \
		_RETURN_FAIL                                                                                                           \
	};

#define DEFAULT_OP_STR_NULL_SN(m_prefix, m_op_name, m_name, m_op, m_type)                                                          \
	CASE_TYPE(m_prefix, m_op_name, m_name) {                                                                                       \
		if (p_b.type == STRING)                                                                                                    \
			_RETURN(*reinterpret_cast<const m_type *>(p_a._data._mem) m_op *reinterpret_cast<const String *>(p_b._data._mem));     \
		if (p_b.type == STRING_NAME)                                                                                               \
			_RETURN(*reinterpret_cast<const m_type *>(p_a._data._mem) m_op *reinterpret_cast<const StringName *>(p_b._data._mem)); \
		if (p_b.type == NIL)                                                                                                       \
			_RETURN(!(p_b.type m_op NIL));                                                                                         \
                                                                                                                                   \
		_RETURN_FAIL                                                                                                               \
	};

#define DEFAULT_OP_LOCALMEM_REV(m_prefix, m_op_name, m_name, m_op, m_type)                                                     \
	CASE_TYPE(m_prefix, m_op_name, m_name) {                                                                                   \
		if (p_b.type == m_name)                                                                                                \
			_RETURN(*reinterpret_cast<const m_type *>(p_b._data._mem) m_op *reinterpret_cast<const m_type *>(p_a._data._mem)); \
                                                                                                                               \
		_RETURN_FAIL                                                                                                           \
	};

#define DEFAULT_OP_LOCALMEM(m_prefix, m_op_name, m_name, m_op, m_type)                                                         \
	CASE_TYPE(m_prefix, m_op_name, m_name) {                                                                                   \
		if (p_b.type == m_name)                                                                                                \
			_RETURN(*reinterpret_cast<const m_type *>(p_a._data._mem) m_op *reinterpret_cast<const m_type *>(p_b._data._mem)); \
                                                                                                                               \
		_RETURN_FAIL                                                                                                           \
	};

#define DEFAULT_OP_LOCALMEM_NULL(m_prefix, m_op_name, m_name, m_op, m_type)                                                    \
	CASE_TYPE(m_prefix, m_op_name, m_name) {                                                                                   \
		if (p_b.type == m_name)                                                                                                \
			_RETURN(*reinterpret_cast<const m_type *>(p_a._data._mem) m_op *reinterpret_cast<const m_type *>(p_b._data._mem)); \
		if (p_b.type == NIL)                                                                                                   \
			_RETURN(!(p_b.type m_op NIL));                                                                                     \
                                                                                                                               \
		_RETURN_FAIL                                                                                                           \
	};

#define DEFAULT_OP_LOCALMEM_NEG(m_prefix, m_op_name, m_name, m_type) \
	CASE_TYPE(m_prefix, m_op_name, m_name) {                         \
		_RETURN(-*reinterpret_cast<const m_type *>(p_a._data._mem)); \
	}

#define DEFAULT_OP_LOCALMEM_POS(m_prefix, m_op_name, m_name, m_type) \
	CASE_TYPE(m_prefix, m_op_name, m_name) {                         \
		_RETURN(*reinterpret_cast<const m_type *>(p_a._data._mem));  \
	}

#define DEFAULT_OP_LOCALMEM_NUM(m_prefix, m_op_name, m_name, m_op, m_type)                                                     \
	CASE_TYPE(m_prefix, m_op_name, m_name) {                                                                                   \
		if (p_b.type == m_name)                                                                                                \
			_RETURN(*reinterpret_cast<const m_type *>(p_a._data._mem) m_op *reinterpret_cast<const m_type *>(p_b._data._mem)); \
		if (p_b.type == INT)                                                                                                   \
			_RETURN(*reinterpret_cast<const m_type *>(p_a._data._mem) m_op p_b._data._int);                                    \
		if (p_b.type == REAL)                                                                                                  \
			_RETURN(*reinterpret_cast<const m_type *>(p_a._data._mem) m_op p_b._data._real);                                   \
                                                                                                                               \
		_RETURN_FAIL                                                                                                           \
	}

#define DEFAULT_OP_PTR(m_op, m_name, m_sub)                \
	CASE_TYPE(m_prefix, m_op_name, m_name) {               \
		if (p_b.type == m_name)                            \
			_RETURN(p_a._data.m_sub m_op p_b._data.m_sub); \
                                                           \
		_RETURN_FAIL                                       \
	}

#define DEFAULT_OP_PTRREF(m_prefix, m_op_name, m_name, m_op, m_sub) \
	CASE_TYPE(m_prefix, m_op_name, m_name) {                        \
		if (p_b.type == m_name)                                     \
			_RETURN(*p_a._data.m_sub m_op *p_b._data.m_sub);        \
                                                                    \
		_RETURN_FAIL                                                \
	}

#define DEFAULT_OP_PTRREF_NULL(m_prefix, m_op_name, m_name, m_op, m_sub) \
	CASE_TYPE(m_prefix, m_op_name, m_name) {                             \
		if (p_b.type == m_name)                                          \
			_RETURN(*p_a._data.m_sub m_op *p_b._data.m_sub);             \
		if (p_b.type == NIL)                                             \
			_RETURN(!(p_b.type m_op NIL));                               \
                                                                         \
		_RETURN_FAIL                                                     \
	}

#define DEFAULT_OP_ARRAY_EQ(m_prefix, m_op_name, m_name, m_type)                                  \
	CASE_TYPE(m_prefix, m_op_name, m_name) {                                                      \
		if (p_b.type == NIL)                                                                      \
			_RETURN(false)                                                                        \
		DEFAULT_OP_ARRAY_OP_BODY(m_prefix, m_op_name, m_name, m_type, !=, !=, true, false, false) \
	}

#define DEFAULT_OP_ARRAY_NEQ(m_prefix, m_op_name, m_name, m_type)                                \
	CASE_TYPE(m_prefix, m_op_name, m_name) {                                                     \
		if (p_b.type == NIL)                                                                     \
			_RETURN(true)                                                                        \
		DEFAULT_OP_ARRAY_OP_BODY(m_prefix, m_op_name, m_name, m_type, !=, !=, false, true, true) \
	}

#define DEFAULT_OP_ARRAY_LT(m_prefix, m_op_name, m_name, m_type) \
	DEFAULT_OP_ARRAY_OP(m_prefix, m_op_name, m_name, m_type, <, !=, false, a_len < array_b.size(), true)

#define DEFAULT_OP_ARRAY_GT(m_prefix, m_op_name, m_name, m_type) \
	DEFAULT_OP_ARRAY_OP(m_prefix, m_op_name, m_name, m_type, >, !=, false, a_len < array_b.size(), true)

#define DEFAULT_OP_ARRAY_OP(m_prefix, m_op_name, m_name, m_type, m_opa, m_opb, m_ret_def, m_ret_s, m_ret_f)      \
	CASE_TYPE(m_prefix, m_op_name, m_name) {                                                                     \
		DEFAULT_OP_ARRAY_OP_BODY(m_prefix, m_op_name, m_name, m_type, m_opa, m_opb, m_ret_def, m_ret_s, m_ret_f) \
	}

#define DEFAULT_OP_ARRAY_OP_BODY(m_prefix, m_op_name, m_name, m_type, m_opa, m_opb, m_ret_def, m_ret_s, m_ret_f) \
	if (p_a.type != p_b.type)                                                                                    \
		_RETURN_FAIL                                                                                             \
                                                                                                                 \
	const PoolVector<m_type> &array_a = *reinterpret_cast<const PoolVector<m_type> *>(p_a._data._mem);           \
	const PoolVector<m_type> &array_b = *reinterpret_cast<const PoolVector<m_type> *>(p_b._data._mem);           \
                                                                                                                 \
	int a_len = array_a.size();                                                                                  \
	if (a_len m_opa array_b.size()) {                                                                            \
		_RETURN(m_ret_s);                                                                                        \
	} else {                                                                                                     \
		PoolVector<m_type>::Read ra = array_a.read();                                                            \
		PoolVector<m_type>::Read rb = array_b.read();                                                            \
                                                                                                                 \
		for (int i = 0; i < a_len; i++) {                                                                        \
			if (ra[i] m_opb rb[i])                                                                               \
				_RETURN(m_ret_f);                                                                                \
		}                                                                                                        \
                                                                                                                 \
		_RETURN(m_ret_def);                                                                                      \
	}

#define DEFAULT_OP_ARRAY_ADD(m_prefix, m_op_name, m_name, m_type)                                          \
	CASE_TYPE(m_prefix, m_op_name, m_name) {                                                               \
		if (p_a.type != p_b.type)                                                                          \
			_RETURN_FAIL;                                                                                  \
                                                                                                           \
		const PoolVector<m_type> &array_a = *reinterpret_cast<const PoolVector<m_type> *>(p_a._data._mem); \
		const PoolVector<m_type> &array_b = *reinterpret_cast<const PoolVector<m_type> *>(p_b._data._mem); \
		PoolVector<m_type> sum = array_a;                                                                  \
		sum.append_array(array_b);                                                                         \
		_RETURN(sum);                                                                                      \
	}

void Variant::evaluate(const Operator &p_op, const Variant &p_a,
		const Variant &p_b, Variant &r_ret, bool &r_valid) {
	CASES(math);
	r_valid = true;

	SWITCH(math, p_op, p_a.type) {
		SWITCH_OP(math, OP_EQUAL, p_a.type) {
			CASE_TYPE(math, OP_EQUAL, NIL) {
				if (p_b.type == NIL)
					_RETURN(true);
				if (p_b.type == OBJECT)
					_RETURN(_UNSAFE_OBJ_PROXY_PTR(p_b) == nullptr);

				_RETURN(false);
			}

			CASE_TYPE(math, OP_EQUAL, BOOL) {
				if (p_b.type != BOOL) {
					if (p_b.type == NIL)
						_RETURN(false);
					_RETURN_FAIL;
				}

				_RETURN(p_a._data._bool == p_b._data._bool);
			}

			CASE_TYPE(math, OP_EQUAL, OBJECT) {
				if (p_b.type == OBJECT)
					_RETURN(_UNSAFE_OBJ_PROXY_PTR(p_a) == _UNSAFE_OBJ_PROXY_PTR(p_b));
				if (p_b.type == NIL)
					_RETURN(_UNSAFE_OBJ_PROXY_PTR(p_a) == nullptr);

				_RETURN_FAIL;
			}

			CASE_TYPE(math, OP_EQUAL, DICTIONARY) {
				if (p_b.type != DICTIONARY) {
					if (p_b.type == NIL)
						_RETURN(false);
					_RETURN_FAIL;
				}

				const Dictionary *arr_a = reinterpret_cast<const Dictionary *>(p_a._data._mem);
				const Dictionary *arr_b = reinterpret_cast<const Dictionary *>(p_b._data._mem);

				_RETURN(*arr_a == *arr_b);
			}

			CASE_TYPE(math, OP_EQUAL, ARRAY) {
				if (p_b.type != ARRAY) {
					if (p_b.type == NIL)
						_RETURN(false);
					_RETURN_FAIL;
				}
				const Array *arr_a = reinterpret_cast<const Array *>(p_a._data._mem);
				const Array *arr_b = reinterpret_cast<const Array *>(p_b._data._mem);

				int l = arr_a->size();
				if (arr_b->size() != l)
					_RETURN(false);
				for (int i = 0; i < l; i++) {
					if (!((*arr_a)[i] == (*arr_b)[i])) {
						_RETURN(false);
					}
				}

				_RETURN(true);
			}

			DEFAULT_OP_NUM_NULL(math, OP_EQUAL, INT, ==, _int);
			DEFAULT_OP_NUM_NULL(math, OP_EQUAL, REAL, ==, _real);
			DEFAULT_OP_STR_NULL(math, OP_EQUAL, STRING, ==, String);
			DEFAULT_OP_LOCALMEM_NULL(math, OP_EQUAL, RECT2, ==, Rect2);
			DEFAULT_OP_LOCALMEM_NULL(math, OP_EQUAL, RECT2I, ==, Rect2i);
			DEFAULT_OP_LOCALMEM_NULL(math, OP_EQUAL, VECTOR2, ==, Vector2);
			DEFAULT_OP_LOCALMEM_NULL(math, OP_EQUAL, VECTOR2I, ==, Vector2i);
			DEFAULT_OP_LOCALMEM_NULL(math, OP_EQUAL, VECTOR3, ==, Vector3);
			DEFAULT_OP_LOCALMEM_NULL(math, OP_EQUAL, VECTOR3I, ==, Vector3i);
			DEFAULT_OP_LOCALMEM_NULL(math, OP_EQUAL, VECTOR4, ==, Vector4);
			DEFAULT_OP_LOCALMEM_NULL(math, OP_EQUAL, VECTOR4I, ==, Vector4i);
			DEFAULT_OP_LOCALMEM_NULL(math, OP_EQUAL, PLANE, ==, Plane);
			DEFAULT_OP_LOCALMEM_NULL(math, OP_EQUAL, QUATERNION, ==, Quaternion);
			DEFAULT_OP_PTRREF_NULL(math, OP_EQUAL, AABB, ==, _aabb);
			DEFAULT_OP_PTRREF_NULL(math, OP_EQUAL, BASIS, ==, _basis);
			DEFAULT_OP_PTRREF_NULL(math, OP_EQUAL, TRANSFORM, ==, _transform);
			DEFAULT_OP_PTRREF_NULL(math, OP_EQUAL, TRANSFORM2D, ==, _transform2d);
			DEFAULT_OP_PTRREF_NULL(math, OP_EQUAL, PROJECTION, ==, _projection);
			DEFAULT_OP_LOCALMEM_NULL(math, OP_EQUAL, COLOR, ==, Color);
			DEFAULT_OP_STR_NULL_SN(math, OP_EQUAL, STRING_NAME, ==, StringName);

			DEFAULT_OP_ARRAY_EQ(math, OP_EQUAL, POOL_BYTE_ARRAY, uint8_t);
			DEFAULT_OP_ARRAY_EQ(math, OP_EQUAL, POOL_INT_ARRAY, int);
			DEFAULT_OP_ARRAY_EQ(math, OP_EQUAL, POOL_REAL_ARRAY, real_t);
			DEFAULT_OP_ARRAY_EQ(math, OP_EQUAL, POOL_STRING_ARRAY, String);
			DEFAULT_OP_ARRAY_EQ(math, OP_EQUAL, POOL_VECTOR2_ARRAY, Vector2);
			DEFAULT_OP_ARRAY_EQ(math, OP_EQUAL, POOL_VECTOR2I_ARRAY, Vector2i);
			DEFAULT_OP_ARRAY_EQ(math, OP_EQUAL, POOL_VECTOR3_ARRAY, Vector3);
			DEFAULT_OP_ARRAY_EQ(math, OP_EQUAL, POOL_VECTOR3I_ARRAY, Vector3i);
			DEFAULT_OP_ARRAY_EQ(math, OP_EQUAL, POOL_VECTOR4_ARRAY, Vector4);
			DEFAULT_OP_ARRAY_EQ(math, OP_EQUAL, POOL_VECTOR4I_ARRAY, Vector4i);
			DEFAULT_OP_ARRAY_EQ(math, OP_EQUAL, POOL_COLOR_ARRAY, Color);
		}

		SWITCH_OP(math, OP_NOT_EQUAL, p_a.type) {
			CASE_TYPE(math, OP_NOT_EQUAL, NIL) {
				if (p_b.type == NIL)
					_RETURN(false);
				if (p_b.type == OBJECT)
					_RETURN(_UNSAFE_OBJ_PROXY_PTR(p_b) != nullptr);

				_RETURN(true);
			}

			CASE_TYPE(math, OP_NOT_EQUAL, BOOL) {
				if (p_b.type != BOOL) {
					if (p_b.type == NIL)
						_RETURN(true);

					_RETURN_FAIL;
				}

				_RETURN(p_a._data._bool != p_b._data._bool);
			}

			CASE_TYPE(math, OP_NOT_EQUAL, OBJECT) {
				if (p_b.type == OBJECT)
					_RETURN((_UNSAFE_OBJ_PROXY_PTR(p_a) != _UNSAFE_OBJ_PROXY_PTR(p_b)));
				if (p_b.type == NIL)
					_RETURN(_UNSAFE_OBJ_PROXY_PTR(p_a) != nullptr);

				_RETURN_FAIL;
			}

			CASE_TYPE(math, OP_NOT_EQUAL, DICTIONARY) {
				if (p_b.type != DICTIONARY) {
					if (p_b.type == NIL)
						_RETURN(true);
					_RETURN_FAIL;
				}

				const Dictionary *arr_a = reinterpret_cast<const Dictionary *>(p_a._data._mem);
				const Dictionary *arr_b = reinterpret_cast<const Dictionary *>(p_b._data._mem);

				_RETURN(*arr_a != *arr_b);
			}

			CASE_TYPE(math, OP_NOT_EQUAL, ARRAY) {
				if (p_b.type != ARRAY) {
					if (p_b.type == NIL)
						_RETURN(true);

					_RETURN_FAIL;
				}

				const Array *arr_a = reinterpret_cast<const Array *>(p_a._data._mem);
				const Array *arr_b = reinterpret_cast<const Array *>(p_b._data._mem);

				int l = arr_a->size();
				if (arr_b->size() != l)
					_RETURN(true);
				for (int i = 0; i < l; i++) {
					if (((*arr_a)[i] != (*arr_b)[i])) {
						_RETURN(true);
					}
				}

				_RETURN(false);
			}

			DEFAULT_OP_NUM_NULL(math, OP_NOT_EQUAL, INT, !=, _int);
			DEFAULT_OP_NUM_NULL(math, OP_NOT_EQUAL, REAL, !=, _real);
			DEFAULT_OP_STR_NULL(math, OP_NOT_EQUAL, STRING, !=, String);
			DEFAULT_OP_LOCALMEM_NULL(math, OP_NOT_EQUAL, RECT2, !=, Rect2);
			DEFAULT_OP_LOCALMEM_NULL(math, OP_NOT_EQUAL, RECT2I, !=, Rect2i);
			DEFAULT_OP_LOCALMEM_NULL(math, OP_NOT_EQUAL, VECTOR2, !=, Vector2);
			DEFAULT_OP_LOCALMEM_NULL(math, OP_NOT_EQUAL, VECTOR2I, !=, Vector2i);
			DEFAULT_OP_LOCALMEM_NULL(math, OP_NOT_EQUAL, VECTOR3, !=, Vector3);
			DEFAULT_OP_LOCALMEM_NULL(math, OP_NOT_EQUAL, VECTOR3I, !=, Vector3i);
			DEFAULT_OP_LOCALMEM_NULL(math, OP_NOT_EQUAL, VECTOR4, !=, Vector4);
			DEFAULT_OP_LOCALMEM_NULL(math, OP_NOT_EQUAL, VECTOR4I, !=, Vector4i);
			DEFAULT_OP_LOCALMEM_NULL(math, OP_NOT_EQUAL, PLANE, !=, Plane);
			DEFAULT_OP_LOCALMEM_NULL(math, OP_NOT_EQUAL, QUATERNION, !=, Quaternion);
			DEFAULT_OP_PTRREF_NULL(math, OP_NOT_EQUAL, AABB, !=, _aabb);
			DEFAULT_OP_PTRREF_NULL(math, OP_NOT_EQUAL, BASIS, !=, _basis);
			DEFAULT_OP_PTRREF_NULL(math, OP_NOT_EQUAL, TRANSFORM, !=, _transform);
			DEFAULT_OP_PTRREF_NULL(math, OP_NOT_EQUAL, TRANSFORM2D, !=, _transform2d);
			DEFAULT_OP_PTRREF_NULL(math, OP_NOT_EQUAL, PROJECTION, !=, _projection);
			DEFAULT_OP_LOCALMEM_NULL(math, OP_NOT_EQUAL, COLOR, !=, Color);
			DEFAULT_OP_STR_NULL_SN(math, OP_NOT_EQUAL, STRING_NAME, !=, StringName);

			DEFAULT_OP_ARRAY_NEQ(math, OP_NOT_EQUAL, POOL_BYTE_ARRAY, uint8_t);
			DEFAULT_OP_ARRAY_NEQ(math, OP_NOT_EQUAL, POOL_INT_ARRAY, int);
			DEFAULT_OP_ARRAY_NEQ(math, OP_NOT_EQUAL, POOL_REAL_ARRAY, real_t);
			DEFAULT_OP_ARRAY_NEQ(math, OP_NOT_EQUAL, POOL_STRING_ARRAY, String);
			DEFAULT_OP_ARRAY_NEQ(math, OP_NOT_EQUAL, POOL_VECTOR2_ARRAY, Vector2);
			DEFAULT_OP_ARRAY_NEQ(math, OP_NOT_EQUAL, POOL_VECTOR2I_ARRAY, Vector2i);
			DEFAULT_OP_ARRAY_NEQ(math, OP_NOT_EQUAL, POOL_VECTOR3_ARRAY, Vector3);
			DEFAULT_OP_ARRAY_NEQ(math, OP_NOT_EQUAL, POOL_VECTOR3I_ARRAY, Vector3i);
			DEFAULT_OP_ARRAY_NEQ(math, OP_NOT_EQUAL, POOL_VECTOR4_ARRAY, Vector4);
			DEFAULT_OP_ARRAY_NEQ(math, OP_NOT_EQUAL, POOL_VECTOR4I_ARRAY, Vector4i);
			DEFAULT_OP_ARRAY_NEQ(math, OP_NOT_EQUAL, POOL_COLOR_ARRAY, Color);
		}

		SWITCH_OP(math, OP_LESS, p_a.type) {
			CASE_TYPE(math, OP_LESS, BOOL) {
				if (p_b.type != BOOL)
					_RETURN_FAIL;

				if (p_a._data._bool == p_b._data._bool)
					_RETURN(false);

				if (p_a._data._bool && !p_b._data._bool)
					_RETURN(false);

				_RETURN(true);
			}

			CASE_TYPE(math, OP_LESS, OBJECT) {
				if (p_b.type != OBJECT)
					_RETURN_FAIL;
				_RETURN(_UNSAFE_OBJ_PROXY_PTR(p_a) < _UNSAFE_OBJ_PROXY_PTR(p_b));
			}

			CASE_TYPE(math, OP_LESS, ARRAY) {
				if (p_b.type != ARRAY)
					_RETURN_FAIL;

				const Array *arr_a = reinterpret_cast<const Array *>(p_a._data._mem);
				const Array *arr_b = reinterpret_cast<const Array *>(p_b._data._mem);

				int l = arr_a->size();
				if (arr_b->size() < l)
					_RETURN(false);
				for (int i = 0; i < l; i++) {
					if (!((*arr_a)[i] < (*arr_b)[i])) {
						_RETURN(true);
					}
				}

				_RETURN(false);
			}

			DEFAULT_OP_NUM(math, OP_LESS, INT, <, _int);
			DEFAULT_OP_NUM(math, OP_LESS, REAL, <, _real);
			DEFAULT_OP_STR(math, OP_LESS, STRING, <, String);
			DEFAULT_OP_LOCALMEM(math, OP_LESS, VECTOR2, <, Vector2);
			DEFAULT_OP_LOCALMEM(math, OP_LESS, VECTOR2I, <, Vector2i);
			DEFAULT_OP_LOCALMEM(math, OP_LESS, VECTOR3, <, Vector3);
			DEFAULT_OP_LOCALMEM(math, OP_LESS, VECTOR3I, <, Vector3i);
			DEFAULT_OP_LOCALMEM(math, OP_LESS, VECTOR4, <, Vector4);
			DEFAULT_OP_LOCALMEM(math, OP_LESS, VECTOR4I, <, Vector4i);

			DEFAULT_OP_ARRAY_LT(math, OP_LESS, POOL_BYTE_ARRAY, uint8_t);
			DEFAULT_OP_ARRAY_LT(math, OP_LESS, POOL_INT_ARRAY, int);
			DEFAULT_OP_ARRAY_LT(math, OP_LESS, POOL_REAL_ARRAY, real_t);
			DEFAULT_OP_ARRAY_LT(math, OP_LESS, POOL_STRING_ARRAY, String);
			DEFAULT_OP_ARRAY_LT(math, OP_LESS, POOL_VECTOR2_ARRAY, Vector2);
			DEFAULT_OP_ARRAY_LT(math, OP_LESS, POOL_VECTOR2I_ARRAY, Vector2i);
			DEFAULT_OP_ARRAY_LT(math, OP_LESS, POOL_VECTOR3_ARRAY, Vector3);
			DEFAULT_OP_ARRAY_LT(math, OP_LESS, POOL_VECTOR3I_ARRAY, Vector3i);
			DEFAULT_OP_ARRAY_LT(math, OP_LESS, POOL_VECTOR4_ARRAY, Vector4);
			DEFAULT_OP_ARRAY_LT(math, OP_LESS, POOL_VECTOR4I_ARRAY, Vector4i);
			DEFAULT_OP_ARRAY_LT(math, OP_LESS, POOL_COLOR_ARRAY, Color);

			CASE_TYPE(math, OP_LESS, NIL)
			CASE_TYPE(math, OP_LESS, RECT2)
			CASE_TYPE(math, OP_LESS, RECT2I)
			CASE_TYPE(math, OP_LESS, PLANE)
			CASE_TYPE(math, OP_LESS, QUATERNION)
			CASE_TYPE(math, OP_LESS, AABB)
			CASE_TYPE(math, OP_LESS, BASIS)
			CASE_TYPE(math, OP_LESS, TRANSFORM)
			CASE_TYPE(math, OP_LESS, TRANSFORM2D)
			CASE_TYPE(math, OP_LESS, PROJECTION)
			CASE_TYPE(math, OP_LESS, STRING_NAME)
			CASE_TYPE(math, OP_LESS, COLOR)
			CASE_TYPE(math, OP_LESS, DICTIONARY)
			_RETURN_FAIL;
		}

		SWITCH_OP(math, OP_LESS_EQUAL, p_a.type) {
			CASE_TYPE(math, OP_LESS_EQUAL, OBJECT) {
				if (p_b.type != OBJECT)
					_RETURN_FAIL;
				_RETURN(_UNSAFE_OBJ_PROXY_PTR(p_a) <= _UNSAFE_OBJ_PROXY_PTR(p_b));
			}

			DEFAULT_OP_NUM(math, OP_LESS_EQUAL, INT, <=, _int);
			DEFAULT_OP_NUM(math, OP_LESS_EQUAL, REAL, <=, _real);
			DEFAULT_OP_STR(math, OP_LESS_EQUAL, STRING, <=, String);
			DEFAULT_OP_LOCALMEM(math, OP_LESS_EQUAL, VECTOR2, <=, Vector2);
			DEFAULT_OP_LOCALMEM(math, OP_LESS_EQUAL, VECTOR2I, <=, Vector2i);
			DEFAULT_OP_LOCALMEM(math, OP_LESS_EQUAL, VECTOR3, <=, Vector3);
			DEFAULT_OP_LOCALMEM(math, OP_LESS_EQUAL, VECTOR3I, <=, Vector3i);
			DEFAULT_OP_LOCALMEM(math, OP_LESS_EQUAL, VECTOR4, <=, Vector4);
			DEFAULT_OP_LOCALMEM(math, OP_LESS_EQUAL, VECTOR4I, <=, Vector4i);

			CASE_TYPE(math, OP_LESS_EQUAL, NIL)
			CASE_TYPE(math, OP_LESS_EQUAL, BOOL)
			CASE_TYPE(math, OP_LESS_EQUAL, RECT2)
			CASE_TYPE(math, OP_LESS_EQUAL, RECT2I)
			CASE_TYPE(math, OP_LESS_EQUAL, PLANE)
			CASE_TYPE(math, OP_LESS_EQUAL, QUATERNION)
			CASE_TYPE(math, OP_LESS_EQUAL, AABB)
			CASE_TYPE(math, OP_LESS_EQUAL, BASIS)
			CASE_TYPE(math, OP_LESS_EQUAL, TRANSFORM)
			CASE_TYPE(math, OP_LESS_EQUAL, TRANSFORM2D)
			CASE_TYPE(math, OP_LESS_EQUAL, PROJECTION)
			CASE_TYPE(math, OP_LESS_EQUAL, COLOR)
			CASE_TYPE(math, OP_LESS_EQUAL, STRING_NAME)
			CASE_TYPE(math, OP_LESS_EQUAL, DICTIONARY)
			CASE_TYPE(math, OP_LESS_EQUAL, ARRAY)
			CASE_TYPE(math, OP_LESS_EQUAL, POOL_BYTE_ARRAY);
			CASE_TYPE(math, OP_LESS_EQUAL, POOL_INT_ARRAY);
			CASE_TYPE(math, OP_LESS_EQUAL, POOL_REAL_ARRAY);
			CASE_TYPE(math, OP_LESS_EQUAL, POOL_STRING_ARRAY);
			CASE_TYPE(math, OP_LESS_EQUAL, POOL_VECTOR2_ARRAY);
			CASE_TYPE(math, OP_LESS_EQUAL, POOL_VECTOR2I_ARRAY);
			CASE_TYPE(math, OP_LESS_EQUAL, POOL_VECTOR3_ARRAY);
			CASE_TYPE(math, OP_LESS_EQUAL, POOL_VECTOR3I_ARRAY);
			CASE_TYPE(math, OP_LESS_EQUAL, POOL_VECTOR4_ARRAY);
			CASE_TYPE(math, OP_LESS_EQUAL, POOL_VECTOR4I_ARRAY);
			CASE_TYPE(math, OP_LESS_EQUAL, POOL_COLOR_ARRAY);
			_RETURN_FAIL;
		}

		SWITCH_OP(math, OP_GREATER, p_a.type) {
			CASE_TYPE(math, OP_GREATER, BOOL) {
				if (p_b.type != BOOL)
					_RETURN_FAIL;

				if (p_a._data._bool == p_b._data._bool)
					_RETURN(false);

				if (!p_a._data._bool && p_b._data._bool)
					_RETURN(false);

				_RETURN(true);
			}

			CASE_TYPE(math, OP_GREATER, OBJECT) {
				if (p_b.type != OBJECT)
					_RETURN_FAIL;
				_RETURN(_UNSAFE_OBJ_PROXY_PTR(p_a) > _UNSAFE_OBJ_PROXY_PTR(p_b));
			}

			CASE_TYPE(math, OP_GREATER, ARRAY) {
				if (p_b.type != ARRAY)
					_RETURN_FAIL;

				const Array *arr_a = reinterpret_cast<const Array *>(p_a._data._mem);
				const Array *arr_b = reinterpret_cast<const Array *>(p_b._data._mem);

				int l = arr_a->size();
				if (arr_b->size() > l)
					_RETURN(false);
				for (int i = 0; i < l; i++) {
					if (((*arr_a)[i] < (*arr_b)[i])) {
						_RETURN(false);
					}
				}

				_RETURN(true);
			}

			DEFAULT_OP_NUM(math, OP_GREATER, INT, >, _int);
			DEFAULT_OP_NUM(math, OP_GREATER, REAL, >, _real);
			DEFAULT_OP_STR_REV(math, OP_GREATER, STRING, <, String);
			DEFAULT_OP_LOCALMEM_REV(math, OP_GREATER, VECTOR2, <, Vector2);
			DEFAULT_OP_LOCALMEM_REV(math, OP_GREATER, VECTOR2I, <, Vector2i);
			DEFAULT_OP_LOCALMEM_REV(math, OP_GREATER, VECTOR3, <, Vector3);
			DEFAULT_OP_LOCALMEM_REV(math, OP_GREATER, VECTOR3I, <, Vector3i);
			DEFAULT_OP_LOCALMEM_REV(math, OP_GREATER, VECTOR4, <, Vector4);
			DEFAULT_OP_LOCALMEM_REV(math, OP_GREATER, VECTOR4I, <, Vector4i);
			DEFAULT_OP_ARRAY_GT(math, OP_GREATER, POOL_BYTE_ARRAY, uint8_t);
			DEFAULT_OP_ARRAY_GT(math, OP_GREATER, POOL_INT_ARRAY, int);
			DEFAULT_OP_ARRAY_GT(math, OP_GREATER, POOL_REAL_ARRAY, real_t);
			DEFAULT_OP_ARRAY_GT(math, OP_GREATER, POOL_STRING_ARRAY, String);
			DEFAULT_OP_ARRAY_GT(math, OP_GREATER, POOL_VECTOR2_ARRAY, Vector2);
			DEFAULT_OP_ARRAY_GT(math, OP_GREATER, POOL_VECTOR2I_ARRAY, Vector2i);
			DEFAULT_OP_ARRAY_GT(math, OP_GREATER, POOL_VECTOR3_ARRAY, Vector3);
			DEFAULT_OP_ARRAY_GT(math, OP_GREATER, POOL_VECTOR3I_ARRAY, Vector3i);
			DEFAULT_OP_ARRAY_GT(math, OP_GREATER, POOL_VECTOR4_ARRAY, Vector4);
			DEFAULT_OP_ARRAY_GT(math, OP_GREATER, POOL_VECTOR4I_ARRAY, Vector4i);
			DEFAULT_OP_ARRAY_GT(math, OP_GREATER, POOL_COLOR_ARRAY, Color);

			CASE_TYPE(math, OP_GREATER, NIL)
			CASE_TYPE(math, OP_GREATER, RECT2)
			CASE_TYPE(math, OP_GREATER, RECT2I)
			CASE_TYPE(math, OP_GREATER, PLANE)
			CASE_TYPE(math, OP_GREATER, QUATERNION)
			CASE_TYPE(math, OP_GREATER, AABB)
			CASE_TYPE(math, OP_GREATER, BASIS)
			CASE_TYPE(math, OP_GREATER, STRING_NAME)
			CASE_TYPE(math, OP_GREATER, TRANSFORM)
			CASE_TYPE(math, OP_GREATER, TRANSFORM2D)
			CASE_TYPE(math, OP_GREATER, PROJECTION)
			CASE_TYPE(math, OP_GREATER, COLOR)
			CASE_TYPE(math, OP_GREATER, DICTIONARY)
			_RETURN_FAIL;
		}

		SWITCH_OP(math, OP_GREATER_EQUAL, p_a.type) {
			CASE_TYPE(math, OP_GREATER_EQUAL, OBJECT) {
				if (p_b.type != OBJECT)
					_RETURN_FAIL;
				_RETURN(_UNSAFE_OBJ_PROXY_PTR(p_a) >= _UNSAFE_OBJ_PROXY_PTR(p_b));
			}

			DEFAULT_OP_NUM(math, OP_GREATER_EQUAL, INT, >=, _int);
			DEFAULT_OP_NUM(math, OP_GREATER_EQUAL, REAL, >=, _real);
			DEFAULT_OP_STR_REV(math, OP_GREATER_EQUAL, STRING, <=, String);
			DEFAULT_OP_LOCALMEM_REV(math, OP_GREATER_EQUAL, VECTOR2, <=, Vector2);
			DEFAULT_OP_LOCALMEM_REV(math, OP_GREATER_EQUAL, VECTOR2I, <=, Vector2i);
			DEFAULT_OP_LOCALMEM_REV(math, OP_GREATER_EQUAL, VECTOR3, <=, Vector3);
			DEFAULT_OP_LOCALMEM_REV(math, OP_GREATER_EQUAL, VECTOR3I, <=, Vector3i);
			DEFAULT_OP_LOCALMEM_REV(math, OP_GREATER_EQUAL, VECTOR4, <=, Vector4);
			DEFAULT_OP_LOCALMEM_REV(math, OP_GREATER_EQUAL, VECTOR4I, <=, Vector4i);

			CASE_TYPE(math, OP_GREATER_EQUAL, NIL)
			CASE_TYPE(math, OP_GREATER_EQUAL, BOOL)
			CASE_TYPE(math, OP_GREATER_EQUAL, RECT2)
			CASE_TYPE(math, OP_GREATER_EQUAL, RECT2I)
			CASE_TYPE(math, OP_GREATER_EQUAL, PLANE)
			CASE_TYPE(math, OP_GREATER_EQUAL, QUATERNION)
			CASE_TYPE(math, OP_GREATER_EQUAL, AABB)
			CASE_TYPE(math, OP_GREATER_EQUAL, BASIS)
			CASE_TYPE(math, OP_GREATER_EQUAL, TRANSFORM)
			CASE_TYPE(math, OP_GREATER_EQUAL, TRANSFORM2D)
			CASE_TYPE(math, OP_GREATER_EQUAL, PROJECTION)
			CASE_TYPE(math, OP_GREATER_EQUAL, COLOR)
			CASE_TYPE(math, OP_GREATER_EQUAL, DICTIONARY)
			CASE_TYPE(math, OP_GREATER_EQUAL, STRING_NAME)
			CASE_TYPE(math, OP_GREATER_EQUAL, ARRAY)
			CASE_TYPE(math, OP_GREATER_EQUAL, POOL_BYTE_ARRAY);
			CASE_TYPE(math, OP_GREATER_EQUAL, POOL_INT_ARRAY);
			CASE_TYPE(math, OP_GREATER_EQUAL, POOL_REAL_ARRAY);
			CASE_TYPE(math, OP_GREATER_EQUAL, POOL_STRING_ARRAY);
			CASE_TYPE(math, OP_GREATER_EQUAL, POOL_VECTOR2_ARRAY);
			CASE_TYPE(math, OP_GREATER_EQUAL, POOL_VECTOR2I_ARRAY);
			CASE_TYPE(math, OP_GREATER_EQUAL, POOL_VECTOR3_ARRAY);
			CASE_TYPE(math, OP_GREATER_EQUAL, POOL_VECTOR3I_ARRAY);
			CASE_TYPE(math, OP_GREATER_EQUAL, POOL_VECTOR4_ARRAY);
			CASE_TYPE(math, OP_GREATER_EQUAL, POOL_VECTOR4I_ARRAY);
			CASE_TYPE(math, OP_GREATER_EQUAL, POOL_COLOR_ARRAY);
			_RETURN_FAIL;
		}

		SWITCH_OP(math, OP_ADD, p_a.type) {
			CASE_TYPE(math, OP_ADD, ARRAY) {
				if (p_a.type != p_b.type)
					_RETURN_FAIL;

				const Array &array_a = *reinterpret_cast<const Array *>(p_a._data._mem);
				const Array &array_b = *reinterpret_cast<const Array *>(p_b._data._mem);
				Array sum;
				int asize = array_a.size();
				int bsize = array_b.size();
				sum.resize(asize + bsize);
				for (int i = 0; i < asize; i++) {
					sum[i] = array_a[i];
				}
				for (int i = 0; i < bsize; i++) {
					sum[i + asize] = array_b[i];
				}
				_RETURN(sum);
			}

			DEFAULT_OP_NUM(math, OP_ADD, INT, +, _int);
			DEFAULT_OP_NUM(math, OP_ADD, REAL, +, _real);
			DEFAULT_OP_STR(math, OP_ADD, STRING, +, String);
			DEFAULT_OP_LOCALMEM(math, OP_ADD, VECTOR2, +, Vector2);
			DEFAULT_OP_LOCALMEM(math, OP_ADD, VECTOR2I, +, Vector2i);
			DEFAULT_OP_LOCALMEM(math, OP_ADD, VECTOR3, +, Vector3);
			DEFAULT_OP_LOCALMEM(math, OP_ADD, VECTOR3I, +, Vector3i);
			DEFAULT_OP_LOCALMEM(math, OP_ADD, VECTOR4, +, Vector4);
			DEFAULT_OP_LOCALMEM(math, OP_ADD, VECTOR4I, +, Vector4i);
			DEFAULT_OP_LOCALMEM(math, OP_ADD, QUATERNION, +, Quaternion);
			DEFAULT_OP_LOCALMEM(math, OP_ADD, COLOR, +, Color);

			DEFAULT_OP_ARRAY_ADD(math, OP_ADD, POOL_BYTE_ARRAY, uint8_t);
			DEFAULT_OP_ARRAY_ADD(math, OP_ADD, POOL_INT_ARRAY, int);
			DEFAULT_OP_ARRAY_ADD(math, OP_ADD, POOL_REAL_ARRAY, real_t);
			DEFAULT_OP_ARRAY_ADD(math, OP_ADD, POOL_STRING_ARRAY, String);
			DEFAULT_OP_ARRAY_ADD(math, OP_ADD, POOL_VECTOR2_ARRAY, Vector2);
			DEFAULT_OP_ARRAY_ADD(math, OP_ADD, POOL_VECTOR2I_ARRAY, Vector2i);
			DEFAULT_OP_ARRAY_ADD(math, OP_ADD, POOL_VECTOR3_ARRAY, Vector3);
			DEFAULT_OP_ARRAY_ADD(math, OP_ADD, POOL_VECTOR3I_ARRAY, Vector3i);
			DEFAULT_OP_ARRAY_ADD(math, OP_ADD, POOL_VECTOR4_ARRAY, Vector4);
			DEFAULT_OP_ARRAY_ADD(math, OP_ADD, POOL_VECTOR4I_ARRAY, Vector4i);
			DEFAULT_OP_ARRAY_ADD(math, OP_ADD, POOL_COLOR_ARRAY, Color);

			CASE_TYPE(math, OP_ADD, NIL)
			CASE_TYPE(math, OP_ADD, BOOL)
			CASE_TYPE(math, OP_ADD, RECT2)
			CASE_TYPE(math, OP_ADD, RECT2I)
			CASE_TYPE(math, OP_ADD, PLANE)
			CASE_TYPE(math, OP_ADD, AABB)
			CASE_TYPE(math, OP_ADD, BASIS)
			CASE_TYPE(math, OP_ADD, TRANSFORM)
			CASE_TYPE(math, OP_ADD, TRANSFORM2D)
			CASE_TYPE(math, OP_ADD, PROJECTION)
			CASE_TYPE(math, OP_ADD, OBJECT)
			CASE_TYPE(math, OP_ADD, DICTIONARY)
			CASE_TYPE(math, OP_ADD, STRING_NAME)
			_RETURN_FAIL;
		}

		SWITCH_OP(math, OP_SUBTRACT, p_a.type) {
			DEFAULT_OP_NUM(math, OP_SUBTRACT, INT, -, _int);
			DEFAULT_OP_NUM(math, OP_SUBTRACT, REAL, -, _real);
			DEFAULT_OP_LOCALMEM(math, OP_SUBTRACT, VECTOR2, -, Vector2);
			DEFAULT_OP_LOCALMEM(math, OP_SUBTRACT, VECTOR2I, -, Vector2i);
			DEFAULT_OP_LOCALMEM(math, OP_SUBTRACT, VECTOR3, -, Vector3);
			DEFAULT_OP_LOCALMEM(math, OP_SUBTRACT, VECTOR3I, -, Vector3i);
			DEFAULT_OP_LOCALMEM(math, OP_SUBTRACT, VECTOR4, -, Vector4);
			DEFAULT_OP_LOCALMEM(math, OP_SUBTRACT, VECTOR4I, -, Vector4i);
			DEFAULT_OP_LOCALMEM(math, OP_SUBTRACT, QUATERNION, -, Quaternion);
			DEFAULT_OP_LOCALMEM(math, OP_SUBTRACT, COLOR, -, Color);

			CASE_TYPE(math, OP_SUBTRACT, NIL)
			CASE_TYPE(math, OP_SUBTRACT, BOOL)
			CASE_TYPE(math, OP_SUBTRACT, STRING)
			CASE_TYPE(math, OP_SUBTRACT, RECT2)
			CASE_TYPE(math, OP_SUBTRACT, RECT2I)
			CASE_TYPE(math, OP_SUBTRACT, PLANE)
			CASE_TYPE(math, OP_SUBTRACT, AABB)
			CASE_TYPE(math, OP_SUBTRACT, BASIS)
			CASE_TYPE(math, OP_SUBTRACT, TRANSFORM)
			CASE_TYPE(math, OP_SUBTRACT, TRANSFORM2D)
			CASE_TYPE(math, OP_SUBTRACT, PROJECTION)
			CASE_TYPE(math, OP_SUBTRACT, OBJECT)
			CASE_TYPE(math, OP_SUBTRACT, STRING_NAME)
			CASE_TYPE(math, OP_SUBTRACT, DICTIONARY)
			CASE_TYPE(math, OP_SUBTRACT, ARRAY)
			CASE_TYPE(math, OP_SUBTRACT, POOL_BYTE_ARRAY);
			CASE_TYPE(math, OP_SUBTRACT, POOL_INT_ARRAY);
			CASE_TYPE(math, OP_SUBTRACT, POOL_REAL_ARRAY);
			CASE_TYPE(math, OP_SUBTRACT, POOL_STRING_ARRAY);
			CASE_TYPE(math, OP_SUBTRACT, POOL_VECTOR2_ARRAY);
			CASE_TYPE(math, OP_SUBTRACT, POOL_VECTOR2I_ARRAY);
			CASE_TYPE(math, OP_SUBTRACT, POOL_VECTOR3_ARRAY);
			CASE_TYPE(math, OP_SUBTRACT, POOL_VECTOR3I_ARRAY);
			CASE_TYPE(math, OP_SUBTRACT, POOL_VECTOR4_ARRAY);
			CASE_TYPE(math, OP_SUBTRACT, POOL_VECTOR4I_ARRAY);
			CASE_TYPE(math, OP_SUBTRACT, POOL_COLOR_ARRAY);
			_RETURN_FAIL;
		}

		SWITCH_OP(math, OP_MULTIPLY, p_a.type) {
			CASE_TYPE(math, OP_MULTIPLY, QUATERNION) {
				switch (p_b.type) {
					case VECTOR3: {
						_RETURN(reinterpret_cast<const Quaternion *>(p_a._data._mem)->xform(*(const Vector3 *)p_b._data._mem));
					}
					case VECTOR3I: {
						_RETURN(reinterpret_cast<const Quaternion *>(p_a._data._mem)->xform(*(const Vector3i *)p_b._data._mem));
					}
					case QUATERNION: {
						_RETURN(*reinterpret_cast<const Quaternion *>(p_a._data._mem) * *reinterpret_cast<const Quaternion *>(p_b._data._mem));
					}
					case REAL: {
						_RETURN(*reinterpret_cast<const Quaternion *>(p_a._data._mem) * p_b._data._real);
					}
					default:
						_RETURN_FAIL;
				}
			}

			CASE_TYPE(math, OP_MULTIPLY, BASIS) {
				switch (p_b.type) {
					case VECTOR3: {
						_RETURN(p_a._data._basis->xform(*(const Vector3 *)p_b._data._mem));
					}
					case VECTOR3I: {
						_RETURN(p_a._data._basis->xform(*(const Vector3i *)p_b._data._mem));
					}
					case BASIS: {
						_RETURN(*p_a._data._basis * *p_b._data._basis);
					}
					default:
						_RETURN_FAIL;
				}
			}

			CASE_TYPE(math, OP_MULTIPLY, TRANSFORM) {
				switch (p_b.type) {
					case VECTOR3: {
						_RETURN(p_a._data._transform->xform(*(const Vector3 *)p_b._data._mem));
					}
					case VECTOR3I: {
						_RETURN(p_a._data._transform->xform(*(const Vector3i *)p_b._data._mem));
					}
					case TRANSFORM: {
						_RETURN(*p_a._data._transform * *p_b._data._transform);
					}
					default:
						_RETURN_FAIL;
				}
			}

			CASE_TYPE(math, OP_MULTIPLY, TRANSFORM2D) {
				switch (p_b.type) {
					case TRANSFORM2D: {
						_RETURN(*p_a._data._transform2d * *p_b._data._transform2d);
					}
					case VECTOR2: {
						_RETURN(p_a._data._transform2d->xform(*(const Vector2 *)p_b._data._mem));
					}
					case VECTOR2I: {
						_RETURN(p_a._data._transform2d->xform(*(const Vector2i *)p_b._data._mem));
					}
					default:
						_RETURN_FAIL;
				}
			}

			CASE_TYPE(math, OP_MULTIPLY, PROJECTION) {
				switch (p_b.type) {
					case VECTOR4: {
						_RETURN(p_a._data._projection->xform(*(const Vector4 *)p_b._data._mem));
					}
					case VECTOR3: {
						_RETURN(p_a._data._projection->xform(*(const Vector3 *)p_b._data._mem));
					}
					case PLANE: {
						_RETURN(p_a._data._projection->xform(*(const Plane *)p_b._data._mem));
					}
					case PROJECTION: {
						_RETURN(p_a._data._projection->operator*(*(const Projection *)p_b._data._mem));
					}
					default:
						_RETURN_FAIL;
				}
			}

			DEFAULT_OP_NUM_VEC(math, OP_MULTIPLY, INT, *, _int);
			DEFAULT_OP_NUM_VEC(math, OP_MULTIPLY, REAL, *, _real);
			DEFAULT_OP_LOCALMEM_NUM(math, OP_MULTIPLY, VECTOR2, *, Vector2);
			DEFAULT_OP_LOCALMEM_NUM(math, OP_MULTIPLY, VECTOR2I, *, Vector2i);
			DEFAULT_OP_LOCALMEM_NUM(math, OP_MULTIPLY, VECTOR3, *, Vector3);
			DEFAULT_OP_LOCALMEM_NUM(math, OP_MULTIPLY, VECTOR3I, *, Vector3i);
			DEFAULT_OP_LOCALMEM_NUM(math, OP_MULTIPLY, VECTOR4, *, Vector4);
			DEFAULT_OP_LOCALMEM_NUM(math, OP_MULTIPLY, VECTOR4I, *, Vector4i);
			DEFAULT_OP_LOCALMEM_NUM(math, OP_MULTIPLY, COLOR, *, Color);

			CASE_TYPE(math, OP_MULTIPLY, NIL)
			CASE_TYPE(math, OP_MULTIPLY, BOOL)
			CASE_TYPE(math, OP_MULTIPLY, STRING)
			CASE_TYPE(math, OP_MULTIPLY, RECT2)
			CASE_TYPE(math, OP_MULTIPLY, RECT2I)
			CASE_TYPE(math, OP_MULTIPLY, PLANE)
			CASE_TYPE(math, OP_MULTIPLY, AABB)
			CASE_TYPE(math, OP_MULTIPLY, OBJECT)
			CASE_TYPE(math, OP_MULTIPLY, STRING_NAME)
			CASE_TYPE(math, OP_MULTIPLY, DICTIONARY)
			CASE_TYPE(math, OP_MULTIPLY, ARRAY)
			CASE_TYPE(math, OP_MULTIPLY, POOL_BYTE_ARRAY);
			CASE_TYPE(math, OP_MULTIPLY, POOL_INT_ARRAY);
			CASE_TYPE(math, OP_MULTIPLY, POOL_REAL_ARRAY);
			CASE_TYPE(math, OP_MULTIPLY, POOL_STRING_ARRAY);
			CASE_TYPE(math, OP_MULTIPLY, POOL_VECTOR2_ARRAY);
			CASE_TYPE(math, OP_MULTIPLY, POOL_VECTOR2I_ARRAY);
			CASE_TYPE(math, OP_MULTIPLY, POOL_VECTOR3_ARRAY);
			CASE_TYPE(math, OP_MULTIPLY, POOL_VECTOR3I_ARRAY);
			CASE_TYPE(math, OP_MULTIPLY, POOL_VECTOR4_ARRAY);
			CASE_TYPE(math, OP_MULTIPLY, POOL_VECTOR4I_ARRAY);
			CASE_TYPE(math, OP_MULTIPLY, POOL_COLOR_ARRAY);
			_RETURN_FAIL;
		}

		SWITCH_OP(math, OP_DIVIDE, p_a.type) {
			CASE_TYPE(math, OP_DIVIDE, QUATERNION) {
				if (p_b.type != REAL)
					_RETURN_FAIL;
#ifdef DEBUG_ENABLED
				if (p_b._data._real == 0) {
					r_valid = false;
					_RETURN("Division By Zero");
				}
#endif
				_RETURN(*reinterpret_cast<const Quaternion *>(p_a._data._mem) / p_b._data._real);
			}

			DEFAULT_OP_NUM_DIV(math, OP_DIVIDE, INT, _int);
			DEFAULT_OP_NUM_DIV(math, OP_DIVIDE, REAL, _real);
			DEFAULT_OP_LOCALMEM_NUM(math, OP_DIVIDE, VECTOR2, /, Vector2);
			DEFAULT_OP_LOCALMEM_NUM(math, OP_DIVIDE, VECTOR2I, /, Vector2i);
			DEFAULT_OP_LOCALMEM_NUM(math, OP_DIVIDE, VECTOR3, /, Vector3);
			DEFAULT_OP_LOCALMEM_NUM(math, OP_DIVIDE, VECTOR3I, /, Vector3i);
			DEFAULT_OP_LOCALMEM_NUM(math, OP_DIVIDE, VECTOR4, /, Vector4);
			DEFAULT_OP_LOCALMEM_NUM(math, OP_DIVIDE, VECTOR4I, /, Vector4i);
			DEFAULT_OP_LOCALMEM_NUM(math, OP_DIVIDE, COLOR, /, Color);

			CASE_TYPE(math, OP_DIVIDE, NIL)
			CASE_TYPE(math, OP_DIVIDE, BOOL)
			CASE_TYPE(math, OP_DIVIDE, STRING)
			CASE_TYPE(math, OP_DIVIDE, RECT2)
			CASE_TYPE(math, OP_DIVIDE, RECT2I)
			CASE_TYPE(math, OP_DIVIDE, PLANE)
			CASE_TYPE(math, OP_DIVIDE, AABB)
			CASE_TYPE(math, OP_DIVIDE, BASIS)
			CASE_TYPE(math, OP_DIVIDE, TRANSFORM)
			CASE_TYPE(math, OP_DIVIDE, TRANSFORM2D)
			CASE_TYPE(math, OP_DIVIDE, PROJECTION)
			CASE_TYPE(math, OP_DIVIDE, OBJECT)
			CASE_TYPE(math, OP_DIVIDE, STRING_NAME)
			CASE_TYPE(math, OP_DIVIDE, DICTIONARY)
			CASE_TYPE(math, OP_DIVIDE, ARRAY)
			CASE_TYPE(math, OP_DIVIDE, POOL_BYTE_ARRAY);
			CASE_TYPE(math, OP_DIVIDE, POOL_INT_ARRAY);
			CASE_TYPE(math, OP_DIVIDE, POOL_REAL_ARRAY);
			CASE_TYPE(math, OP_DIVIDE, POOL_STRING_ARRAY);
			CASE_TYPE(math, OP_DIVIDE, POOL_VECTOR2_ARRAY);
			CASE_TYPE(math, OP_DIVIDE, POOL_VECTOR2I_ARRAY);
			CASE_TYPE(math, OP_DIVIDE, POOL_VECTOR3_ARRAY);
			CASE_TYPE(math, OP_DIVIDE, POOL_VECTOR3I_ARRAY);
			CASE_TYPE(math, OP_DIVIDE, POOL_VECTOR4_ARRAY);
			CASE_TYPE(math, OP_DIVIDE, POOL_VECTOR4I_ARRAY);
			CASE_TYPE(math, OP_DIVIDE, POOL_COLOR_ARRAY);
			_RETURN_FAIL;
		}

		SWITCH_OP(math, OP_POSITIVE, p_a.type) {
			DEFAULT_OP_NUM_POS(math, OP_POSITIVE, INT, _int);
			DEFAULT_OP_NUM_POS(math, OP_POSITIVE, REAL, _real);
			DEFAULT_OP_LOCALMEM_POS(math, OP_POSITIVE, VECTOR2, Vector2);
			DEFAULT_OP_LOCALMEM_POS(math, OP_POSITIVE, VECTOR2I, Vector2i);
			DEFAULT_OP_LOCALMEM_POS(math, OP_POSITIVE, VECTOR3, Vector3);
			DEFAULT_OP_LOCALMEM_POS(math, OP_POSITIVE, VECTOR3I, Vector3i);
			DEFAULT_OP_LOCALMEM_POS(math, OP_POSITIVE, VECTOR4, Vector4);
			DEFAULT_OP_LOCALMEM_POS(math, OP_POSITIVE, VECTOR4I, Vector4i);
			DEFAULT_OP_LOCALMEM_POS(math, OP_POSITIVE, PLANE, Plane);
			DEFAULT_OP_LOCALMEM_POS(math, OP_POSITIVE, QUATERNION, Quaternion);

			CASE_TYPE(math, OP_POSITIVE, NIL)
			CASE_TYPE(math, OP_POSITIVE, BOOL)
			CASE_TYPE(math, OP_POSITIVE, STRING)
			CASE_TYPE(math, OP_POSITIVE, RECT2)
			CASE_TYPE(math, OP_POSITIVE, RECT2I)
			CASE_TYPE(math, OP_POSITIVE, AABB)
			CASE_TYPE(math, OP_POSITIVE, BASIS)
			CASE_TYPE(math, OP_POSITIVE, TRANSFORM)
			CASE_TYPE(math, OP_POSITIVE, TRANSFORM2D)
			CASE_TYPE(math, OP_POSITIVE, PROJECTION)
			CASE_TYPE(math, OP_POSITIVE, COLOR)
			CASE_TYPE(math, OP_POSITIVE, OBJECT)
			CASE_TYPE(math, OP_POSITIVE, STRING_NAME)
			CASE_TYPE(math, OP_POSITIVE, DICTIONARY)
			CASE_TYPE(math, OP_POSITIVE, ARRAY)
			CASE_TYPE(math, OP_POSITIVE, POOL_BYTE_ARRAY)
			CASE_TYPE(math, OP_POSITIVE, POOL_INT_ARRAY)
			CASE_TYPE(math, OP_POSITIVE, POOL_REAL_ARRAY)
			CASE_TYPE(math, OP_POSITIVE, POOL_STRING_ARRAY)
			CASE_TYPE(math, OP_POSITIVE, POOL_VECTOR2_ARRAY)
			CASE_TYPE(math, OP_POSITIVE, POOL_VECTOR2I_ARRAY)
			CASE_TYPE(math, OP_POSITIVE, POOL_VECTOR3_ARRAY)
			CASE_TYPE(math, OP_POSITIVE, POOL_VECTOR3I_ARRAY)
			CASE_TYPE(math, OP_POSITIVE, POOL_VECTOR4_ARRAY)
			CASE_TYPE(math, OP_POSITIVE, POOL_VECTOR4I_ARRAY)
			CASE_TYPE(math, OP_POSITIVE, POOL_COLOR_ARRAY)
			_RETURN_FAIL;
		}

		SWITCH_OP(math, OP_NEGATE, p_a.type) {
			DEFAULT_OP_NUM_NEG(math, OP_NEGATE, INT, _int);
			DEFAULT_OP_NUM_NEG(math, OP_NEGATE, REAL, _real);

			DEFAULT_OP_LOCALMEM_NEG(math, OP_NEGATE, VECTOR2, Vector2);
			DEFAULT_OP_LOCALMEM_NEG(math, OP_NEGATE, VECTOR2I, Vector2i);
			DEFAULT_OP_LOCALMEM_NEG(math, OP_NEGATE, VECTOR3, Vector3);
			DEFAULT_OP_LOCALMEM_NEG(math, OP_NEGATE, VECTOR3I, Vector3i);
			DEFAULT_OP_LOCALMEM_NEG(math, OP_NEGATE, VECTOR4, Vector4);
			DEFAULT_OP_LOCALMEM_NEG(math, OP_NEGATE, VECTOR4I, Vector4i);
			DEFAULT_OP_LOCALMEM_NEG(math, OP_NEGATE, PLANE, Plane);
			DEFAULT_OP_LOCALMEM_NEG(math, OP_NEGATE, QUATERNION, Quaternion);
			DEFAULT_OP_LOCALMEM_NEG(math, OP_NEGATE, COLOR, Color);

			CASE_TYPE(math, OP_NEGATE, NIL)
			CASE_TYPE(math, OP_NEGATE, BOOL)
			CASE_TYPE(math, OP_NEGATE, STRING)
			CASE_TYPE(math, OP_NEGATE, RECT2)
			CASE_TYPE(math, OP_NEGATE, RECT2I)
			CASE_TYPE(math, OP_NEGATE, AABB)
			CASE_TYPE(math, OP_NEGATE, BASIS)
			CASE_TYPE(math, OP_NEGATE, TRANSFORM)
			CASE_TYPE(math, OP_NEGATE, TRANSFORM2D)
			CASE_TYPE(math, OP_NEGATE, PROJECTION)
			CASE_TYPE(math, OP_NEGATE, OBJECT)
			CASE_TYPE(math, OP_NEGATE, STRING_NAME)
			CASE_TYPE(math, OP_NEGATE, DICTIONARY)
			CASE_TYPE(math, OP_NEGATE, ARRAY)
			CASE_TYPE(math, OP_NEGATE, POOL_BYTE_ARRAY)
			CASE_TYPE(math, OP_NEGATE, POOL_INT_ARRAY)
			CASE_TYPE(math, OP_NEGATE, POOL_REAL_ARRAY)
			CASE_TYPE(math, OP_NEGATE, POOL_STRING_ARRAY)
			CASE_TYPE(math, OP_NEGATE, POOL_VECTOR2_ARRAY)
			CASE_TYPE(math, OP_NEGATE, POOL_VECTOR2I_ARRAY)
			CASE_TYPE(math, OP_NEGATE, POOL_VECTOR3_ARRAY)
			CASE_TYPE(math, OP_NEGATE, POOL_VECTOR3I_ARRAY)
			CASE_TYPE(math, OP_NEGATE, POOL_VECTOR4_ARRAY)
			CASE_TYPE(math, OP_NEGATE, POOL_VECTOR4I_ARRAY)
			CASE_TYPE(math, OP_NEGATE, POOL_COLOR_ARRAY)
			_RETURN_FAIL;
		}

		SWITCH_OP(math, OP_MODULE, p_a.type) {
			CASE_TYPE(math, OP_MODULE, INT) {
				if (p_b.type != INT)
					_RETURN_FAIL;
#ifdef DEBUG_ENABLED
				if (p_b._data._int == 0) {
					r_valid = false;
					_RETURN("Division By Zero");
				}
#endif
				_RETURN(p_a._data._int % p_b._data._int);
			}

			CASE_TYPE(math, OP_MODULE, STRING) {
				const String *format = reinterpret_cast<const String *>(p_a._data._mem);

				String result;
				bool error;
				if (p_b.type == ARRAY) {
					// e.g. "frog %s %d" % ["fish", 12]
					const Array *args = reinterpret_cast<const Array *>(p_b._data._mem);
					result = args->sprintf(*format, &error);
				} else {
					// e.g. "frog %d" % 12
					Array args;
					args.push_back(p_b);
					result = args.sprintf(*format, &error);
				}
				r_valid = !error;
				_RETURN(result);
			}

			CASE_TYPE(math, OP_MODULE, NIL)
			CASE_TYPE(math, OP_MODULE, BOOL)
			CASE_TYPE(math, OP_MODULE, REAL)
			CASE_TYPE(math, OP_MODULE, RECT2)
			CASE_TYPE(math, OP_MODULE, RECT2I)
			CASE_TYPE(math, OP_MODULE, VECTOR2)
			CASE_TYPE(math, OP_MODULE, VECTOR2I)
			CASE_TYPE(math, OP_MODULE, VECTOR3)
			CASE_TYPE(math, OP_MODULE, VECTOR3I)
			CASE_TYPE(math, OP_MODULE, VECTOR4)
			CASE_TYPE(math, OP_MODULE, VECTOR4I)
			CASE_TYPE(math, OP_MODULE, PLANE)
			CASE_TYPE(math, OP_MODULE, QUATERNION)
			CASE_TYPE(math, OP_MODULE, AABB)
			CASE_TYPE(math, OP_MODULE, BASIS)
			CASE_TYPE(math, OP_MODULE, TRANSFORM)
			CASE_TYPE(math, OP_MODULE, TRANSFORM2D)
			CASE_TYPE(math, OP_MODULE, PROJECTION)
			CASE_TYPE(math, OP_MODULE, COLOR)
			CASE_TYPE(math, OP_MODULE, OBJECT)
			CASE_TYPE(math, OP_MODULE, STRING_NAME)
			CASE_TYPE(math, OP_MODULE, DICTIONARY)
			CASE_TYPE(math, OP_MODULE, ARRAY)
			CASE_TYPE(math, OP_MODULE, POOL_BYTE_ARRAY)
			CASE_TYPE(math, OP_MODULE, POOL_INT_ARRAY)
			CASE_TYPE(math, OP_MODULE, POOL_REAL_ARRAY)
			CASE_TYPE(math, OP_MODULE, POOL_STRING_ARRAY)
			CASE_TYPE(math, OP_MODULE, POOL_VECTOR2_ARRAY)
			CASE_TYPE(math, OP_MODULE, POOL_VECTOR2I_ARRAY)
			CASE_TYPE(math, OP_MODULE, POOL_VECTOR3_ARRAY)
			CASE_TYPE(math, OP_MODULE, POOL_VECTOR3I_ARRAY)
			CASE_TYPE(math, OP_MODULE, POOL_VECTOR4_ARRAY)
			CASE_TYPE(math, OP_MODULE, POOL_VECTOR4I_ARRAY)
			CASE_TYPE(math, OP_MODULE, POOL_COLOR_ARRAY)
			_RETURN_FAIL;
		}

		SWITCH_OP(math, OP_STRING_CONCAT, p_a.type) {
			CASE_TYPE_ALL(math, OP_STRING_CONCAT)

			_RETURN(p_a.operator String() + p_b.operator String());
		}

		SWITCH_OP(math, OP_SHIFT_LEFT, p_a.type) {
			CASE_TYPE(math, OP_SHIFT_LEFT, INT) {
				if (p_b.type != INT)
					_RETURN_FAIL;
				if (p_b._data._int < 0 || p_b._data._int >= 64)
					_RETURN_FAIL;
				_RETURN(p_a._data._int << p_b._data._int);
			}

			CASE_TYPE_ALL_BUT_INT(math, OP_SHIFT_LEFT)
			_RETURN_FAIL;
		}

		SWITCH_OP(math, OP_SHIFT_RIGHT, p_a.type) {
			CASE_TYPE(math, OP_SHIFT_RIGHT, INT) {
				if (p_b.type != INT)
					_RETURN_FAIL;
				if (p_b._data._int < 0 || p_b._data._int >= 64)
					_RETURN_FAIL;
				_RETURN(p_a._data._int >> p_b._data._int);
			}

			CASE_TYPE_ALL_BUT_INT(math, OP_SHIFT_RIGHT)
			_RETURN_FAIL;
		}

		SWITCH_OP(math, OP_BIT_AND, p_a.type) {
			CASE_TYPE(math, OP_BIT_AND, INT) {
				if (p_b.type != INT)
					_RETURN_FAIL;
				_RETURN(p_a._data._int & p_b._data._int);
			}

			CASE_TYPE_ALL_BUT_INT(math, OP_BIT_AND)
			_RETURN_FAIL;
		}

		SWITCH_OP(math, OP_BIT_OR, p_a.type) {
			CASE_TYPE(math, OP_BIT_OR, INT) {
				if (p_b.type != INT)
					_RETURN_FAIL;
				_RETURN(p_a._data._int | p_b._data._int);
			}

			CASE_TYPE_ALL_BUT_INT(math, OP_BIT_OR)
			_RETURN_FAIL;
		}

		SWITCH_OP(math, OP_BIT_XOR, p_a.type) {
			CASE_TYPE(math, OP_BIT_XOR, INT) {
				if (p_b.type != INT)
					_RETURN_FAIL;
				_RETURN(p_a._data._int ^ p_b._data._int);
			}

			CASE_TYPE_ALL_BUT_INT(math, OP_BIT_XOR)
			_RETURN_FAIL;
		}

		SWITCH_OP(math, OP_BIT_NEGATE, p_a.type) {
			CASE_TYPE(math, OP_BIT_NEGATE, INT) {
				_RETURN(~p_a._data._int);
			}

			CASE_TYPE_ALL_BUT_INT(math, OP_BIT_NEGATE)
			_RETURN_FAIL;
		}

		SWITCH_OP(math, OP_AND, p_a.type) {
			CASE_TYPE_ALL(math, OP_AND) {
				bool l = p_a.booleanize();
				bool r = p_b.booleanize();

				_RETURN(l && r);
			}
		}

		SWITCH_OP(math, OP_OR, p_a.type) {
			CASE_TYPE_ALL(math, OP_OR) {
				bool l = p_a.booleanize();
				bool r = p_b.booleanize();

				_RETURN(l || r);
			}
		}

		SWITCH_OP(math, OP_XOR, p_a.type) {
			CASE_TYPE_ALL(math, OP_XOR) {
				bool l = p_a.booleanize();
				bool r = p_b.booleanize();

				_RETURN((l || r) && !(l && r));
			}
		}

		SWITCH_OP(math, OP_NOT, p_a.type) {
			CASE_TYPE_ALL(math, OP_NOT) {
				bool l = p_a.booleanize();
				_RETURN(!l);
			}
		}

		SWITCH_OP(math, OP_IN, p_a.type) {
			CASE_TYPE_ALL(math, OP_IN)
			_RETURN(p_b.in(p_a, &r_valid));
		}
	}
}

void Variant::set_named(const StringName &p_index, const Variant &p_value, bool *r_valid) {
	bool valid = false;
	switch (type) {
		case RECT2: {
			if (p_value.type == Variant::VECTOR2) {
				Rect2 *v = reinterpret_cast<Rect2 *>(_data._mem);
				//scalar name
				if (p_index == CoreStringNames::singleton->position) {
					v->position = *reinterpret_cast<const Vector2 *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->size) {
					v->size = *reinterpret_cast<const Vector2 *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->end) {
					v->size = *reinterpret_cast<const Vector2 *>(p_value._data._mem) - v->position;
					valid = true;
				}
			} else if (p_value.type == Variant::VECTOR2I) {
				Rect2 *v = reinterpret_cast<Rect2 *>(_data._mem);
				//scalar name
				if (p_index == CoreStringNames::singleton->position) {
					v->position = *reinterpret_cast<const Vector2i *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->size) {
					v->size = *reinterpret_cast<const Vector2i *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->end) {
					v->size = *reinterpret_cast<const Vector2i *>(p_value._data._mem) - v->position;
					valid = true;
				}
			}
		} break;
		case RECT2I: {
			if (p_value.type == Variant::VECTOR2) {
				Rect2i *v = reinterpret_cast<Rect2i *>(_data._mem);
				//scalar name
				if (p_index == CoreStringNames::singleton->position) {
					v->position = *reinterpret_cast<const Vector2 *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->size) {
					v->size = *reinterpret_cast<const Vector2 *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->end) {
					v->size = *reinterpret_cast<const Vector2 *>(p_value._data._mem) - v->position;
					valid = true;
				}
			} else if (p_value.type == Variant::VECTOR2I) {
				Rect2i *v = reinterpret_cast<Rect2i *>(_data._mem);
				//scalar name
				if (p_index == CoreStringNames::singleton->position) {
					v->position = *reinterpret_cast<const Vector2i *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->size) {
					v->size = *reinterpret_cast<const Vector2i *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->end) {
					v->size = *reinterpret_cast<const Vector2i *>(p_value._data._mem) - v->position;
					valid = true;
				}
			}
		} break;
		case VECTOR2: {
			if (p_value.type == Variant::INT) {
				Vector2 *v = reinterpret_cast<Vector2 *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->x = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->y = p_value._data._int;
					valid = true;
				}
			} else if (p_value.type == Variant::REAL) {
				Vector2 *v = reinterpret_cast<Vector2 *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->x = p_value._data._real;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->y = p_value._data._real;
					valid = true;
				}
			}

		} break;
		case VECTOR2I: {
			if (p_value.type == Variant::INT) {
				Vector2i *v = reinterpret_cast<Vector2i *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->x = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->y = p_value._data._int;
					valid = true;
				}
			} else if (p_value.type == Variant::REAL) {
				Vector2i *v = reinterpret_cast<Vector2i *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->x = static_cast<int>(p_value._data._real);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->y = static_cast<int>(p_value._data._real);
					valid = true;
				}
			}

		} break;
		case VECTOR3: {
			if (p_value.type == Variant::INT) {
				Vector3 *v = reinterpret_cast<Vector3 *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->x = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->y = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->z) {
					v->z = p_value._data._int;
					valid = true;
				}
			} else if (p_value.type == Variant::REAL) {
				Vector3 *v = reinterpret_cast<Vector3 *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->x = p_value._data._real;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->y = p_value._data._real;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->z) {
					v->z = p_value._data._real;
					valid = true;
				}
			}

		} break;
		case VECTOR3I: {
			if (p_value.type == Variant::INT) {
				Vector3i *v = reinterpret_cast<Vector3i *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->x = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->y = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->z) {
					v->z = p_value._data._int;
					valid = true;
				}
			} else if (p_value.type == Variant::REAL) {
				Vector3i *v = reinterpret_cast<Vector3i *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->x = p_value._data._real;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->y = p_value._data._real;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->z) {
					v->z = p_value._data._real;
					valid = true;
				}
			}

		} break;
		case VECTOR4: {
			if (p_value.type == Variant::INT) {
				Vector4 *v = reinterpret_cast<Vector4 *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->x = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->y = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->z) {
					v->z = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->w) {
					v->w = p_value._data._int;
					valid = true;
				}
			} else if (p_value.type == Variant::REAL) {
				Vector4 *v = reinterpret_cast<Vector4 *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->x = p_value._data._real;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->y = p_value._data._real;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->z) {
					v->z = p_value._data._real;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->w) {
					v->w = p_value._data._real;
					valid = true;
				}
			}

		} break;
		case VECTOR4I: {
			if (p_value.type == Variant::INT) {
				Vector4i *v = reinterpret_cast<Vector4i *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->x = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->y = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->z) {
					v->z = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->w) {
					v->w = p_value._data._int;
					valid = true;
				}
			} else if (p_value.type == Variant::REAL) {
				Vector4i *v = reinterpret_cast<Vector4i *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->x = p_value._data._real;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->y = p_value._data._real;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->z) {
					v->z = p_value._data._real;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->w) {
					v->w = p_value._data._real;
					valid = true;
				}
			}

		} break;
		case PLANE: {
			if (p_value.type == Variant::INT) {
				Plane *v = reinterpret_cast<Plane *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->normal.x = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->normal.y = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->z) {
					v->normal.z = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->d) {
					v->d = p_value._data._int;
					valid = true;
				}
			} else if (p_value.type == Variant::REAL) {
				Plane *v = reinterpret_cast<Plane *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->normal.x = p_value._data._real;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->normal.y = p_value._data._real;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->z) {
					v->normal.z = p_value._data._real;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->d) {
					v->d = p_value._data._real;
					valid = true;
				}

			} else if (p_value.type == Variant::VECTOR3) {
				Plane *v = reinterpret_cast<Plane *>(_data._mem);
				if (p_index == CoreStringNames::singleton->normal) {
					v->normal = *reinterpret_cast<const Vector3 *>(p_value._data._mem);
					valid = true;
				}
			}

		} break;
		case QUATERNION: {
			if (p_value.type == Variant::INT) {
				Quaternion *v = reinterpret_cast<Quaternion *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->x = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->y = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->z) {
					v->z = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->w) {
					v->w = p_value._data._int;
					valid = true;
				}
			} else if (p_value.type == Variant::REAL) {
				Quaternion *v = reinterpret_cast<Quaternion *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->x = p_value._data._real;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->y = p_value._data._real;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->z) {
					v->z = p_value._data._real;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->w) {
					v->w = p_value._data._real;
					valid = true;
				}
			}

		} break; // 10
		case AABB: {
			if (p_value.type == Variant::VECTOR3) {
				::AABB *v = _data._aabb;
				//scalar name
				if (p_index == CoreStringNames::singleton->position) {
					v->position = *reinterpret_cast<const Vector3 *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->size) {
					v->size = *reinterpret_cast<const Vector3 *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->end) {
					v->size = *reinterpret_cast<const Vector3 *>(p_value._data._mem) - v->position;
					valid = true;
				}
			} else if (p_value.type == Variant::VECTOR3I) {
				::AABB *v = _data._aabb;
				//scalar name
				if (p_index == CoreStringNames::singleton->position) {
					v->position = *reinterpret_cast<const Vector3i *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->size) {
					v->size = *reinterpret_cast<const Vector3i *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->end) {
					v->size = Vector3(*reinterpret_cast<const Vector3i *>(p_value._data._mem)) - v->position;
					valid = true;
				}
			}
		} break;
		case BASIS: {
			if (p_value.type == Variant::VECTOR3) {
				Basis *v = _data._basis;
				//scalar name
				if (p_index == CoreStringNames::singleton->x) {
					v->set_axis(0, *reinterpret_cast<const Vector3 *>(p_value._data._mem));
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->set_axis(1, *reinterpret_cast<const Vector3 *>(p_value._data._mem));
					valid = true;
				} else if (p_index == CoreStringNames::singleton->z) {
					v->set_axis(2, *reinterpret_cast<const Vector3 *>(p_value._data._mem));
					valid = true;
				}
			} else if (p_value.type == Variant::VECTOR3I) {
				Basis *v = _data._basis;
				//scalar name
				if (p_index == CoreStringNames::singleton->x) {
					v->set_axis(0, *reinterpret_cast<const Vector3i *>(p_value._data._mem));
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->set_axis(1, *reinterpret_cast<const Vector3i *>(p_value._data._mem));
					valid = true;
				} else if (p_index == CoreStringNames::singleton->z) {
					v->set_axis(2, *reinterpret_cast<const Vector3i *>(p_value._data._mem));
					valid = true;
				}
			}
		} break;
		case TRANSFORM: {
			if (p_value.type == Variant::BASIS && p_index == CoreStringNames::singleton->basis) {
				_data._transform->basis = *p_value._data._basis;
				valid = true;
			} else if (p_value.type == Variant::VECTOR3 && p_index == CoreStringNames::singleton->origin) {
				_data._transform->origin = *reinterpret_cast<const Vector3 *>(p_value._data._mem);
				valid = true;
			} else if (p_value.type == Variant::VECTOR3I && p_index == CoreStringNames::singleton->origin) {
				_data._transform->origin = *reinterpret_cast<const Vector3i *>(p_value._data._mem);
				valid = true;
			}

		} break;
		case TRANSFORM2D: {
			if (p_value.type == Variant::VECTOR2) {
				Transform2D *v = _data._transform2d;
				if (p_index == CoreStringNames::singleton->x) {
					v->columns[0] = *reinterpret_cast<const Vector2 *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->columns[1] = *reinterpret_cast<const Vector2 *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->origin) {
					v->columns[2] = *reinterpret_cast<const Vector2 *>(p_value._data._mem);
					valid = true;
				}
			} else if (p_value.type == Variant::VECTOR2I) {
				Transform2D *v = _data._transform2d;
				if (p_index == CoreStringNames::singleton->x) {
					v->columns[0] = *reinterpret_cast<const Vector2i *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->columns[1] = *reinterpret_cast<const Vector2i *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->origin) {
					v->columns[2] = *reinterpret_cast<const Vector2i *>(p_value._data._mem);
					valid = true;
				}
			}

		} break;
		case PROJECTION: {
			if (p_value.type == Variant::VECTOR4) {
				Projection *v = _data._projection;
				if (p_index == CoreStringNames::singleton->x) {
					v->matrix[0] = *reinterpret_cast<const Vector4 *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->matrix[1] = *reinterpret_cast<const Vector4 *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->z) {
					v->matrix[2] = *reinterpret_cast<const Vector4 *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->w) {
					v->matrix[3] = *reinterpret_cast<const Vector4 *>(p_value._data._mem);
					valid = true;
				}
			}

		} break;
		case COLOR: {
			if (p_value.type == Variant::INT) {
				Color *v = reinterpret_cast<Color *>(_data._mem);
				if (p_index == CoreStringNames::singleton->r) {
					v->r = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->g) {
					v->g = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->b) {
					v->b = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->a) {
					v->a = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->r8) {
					v->r = p_value._data._int / 255.0;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->g8) {
					v->g = p_value._data._int / 255.0;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->b8) {
					v->b = p_value._data._int / 255.0;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->a8) {
					v->a = p_value._data._int / 255.0;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->h) {
					v->set_hsv(p_value._data._int, v->get_s(), v->get_v(), v->a);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->s) {
					v->set_hsv(v->get_h(), p_value._data._int, v->get_v(), v->a);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->v) {
					v->set_hsv(v->get_h(), v->get_s(), p_value._data._int, v->a);
					valid = true;
				}
			} else if (p_value.type == Variant::REAL) {
				Color *v = reinterpret_cast<Color *>(_data._mem);
				if (p_index == CoreStringNames::singleton->r) {
					v->r = p_value._data._real;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->g) {
					v->g = p_value._data._real;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->b) {
					v->b = p_value._data._real;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->a) {
					v->a = p_value._data._real;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->r8) {
					v->r = p_value._data._real / 255.0;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->g8) {
					v->g = p_value._data._real / 255.0;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->b8) {
					v->b = p_value._data._real / 255.0;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->a8) {
					v->a = p_value._data._real / 255.0;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->h) {
					v->set_hsv(p_value._data._real, v->get_s(), v->get_v(), v->a);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->s) {
					v->set_hsv(v->get_h(), p_value._data._real, v->get_v(), v->a);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->v) {
					v->set_hsv(v->get_h(), v->get_s(), p_value._data._real, v->a);
					valid = true;
				}
			}
		} break;
		case OBJECT: {
			Object *obj = _OBJ_PTR(*this);
			if (unlikely(!obj)) {
#ifdef DEBUG_ENABLED
				if (_get_obj().rc) {
					ERR_PRINT("Attempted set on a deleted object.");
				}
#endif
				break;
			}
			obj->set(p_index, p_value, &valid);

		} break;
		default: {
			set(p_index.operator String(), p_value, &valid);
		} break;
	}

	if (r_valid) {
		*r_valid = valid;
	}
}

Variant Variant::get_named(const StringName &p_index, bool *r_valid) const {
	if (r_valid) {
		*r_valid = true;
	}
	switch (type) {
		case RECT2: {
			const Rect2 *v = reinterpret_cast<const Rect2 *>(_data._mem);
			//scalar name
			if (p_index == CoreStringNames::singleton->position) {
				return v->position;
			} else if (p_index == CoreStringNames::singleton->size) {
				return v->size;
			} else if (p_index == CoreStringNames::singleton->end) {
				return v->size + v->position;
			}
		} break;
		case RECT2I: {
			const Rect2i *v = reinterpret_cast<const Rect2i *>(_data._mem);
			//scalar name
			if (p_index == CoreStringNames::singleton->position) {
				return v->position;
			} else if (p_index == CoreStringNames::singleton->size) {
				return v->size;
			} else if (p_index == CoreStringNames::singleton->end) {
				return v->size + v->position;
			}
		} break;
		case VECTOR2: {
			const Vector2 *v = reinterpret_cast<const Vector2 *>(_data._mem);
			if (p_index == CoreStringNames::singleton->x) {
				return v->x;
			} else if (p_index == CoreStringNames::singleton->y) {
				return v->y;
			}

		} break;
		case VECTOR2I: {
			const Vector2i *v = reinterpret_cast<const Vector2i *>(_data._mem);
			if (p_index == CoreStringNames::singleton->x) {
				return v->x;
			} else if (p_index == CoreStringNames::singleton->y) {
				return v->y;
			}

		} break;
		case VECTOR3: {
			const Vector3 *v = reinterpret_cast<const Vector3 *>(_data._mem);
			if (p_index == CoreStringNames::singleton->x) {
				return v->x;
			} else if (p_index == CoreStringNames::singleton->y) {
				return v->y;
			} else if (p_index == CoreStringNames::singleton->z) {
				return v->z;
			}

		} break;
		case VECTOR3I: {
			const Vector3i *v = reinterpret_cast<const Vector3i *>(_data._mem);
			if (p_index == CoreStringNames::singleton->x) {
				return v->x;
			} else if (p_index == CoreStringNames::singleton->y) {
				return v->y;
			} else if (p_index == CoreStringNames::singleton->z) {
				return v->z;
			}

		} break;
		case VECTOR4: {
			const Vector4 *v = reinterpret_cast<const Vector4 *>(_data._mem);
			if (p_index == CoreStringNames::singleton->x) {
				return v->x;
			} else if (p_index == CoreStringNames::singleton->y) {
				return v->y;
			} else if (p_index == CoreStringNames::singleton->z) {
				return v->z;
			} else if (p_index == CoreStringNames::singleton->w) {
				return v->w;
			}

		} break;
		case VECTOR4I: {
			const Vector4i *v = reinterpret_cast<const Vector4i *>(_data._mem);
			if (p_index == CoreStringNames::singleton->x) {
				return v->x;
			} else if (p_index == CoreStringNames::singleton->y) {
				return v->y;
			} else if (p_index == CoreStringNames::singleton->z) {
				return v->z;
			} else if (p_index == CoreStringNames::singleton->w) {
				return v->w;
			}

		} break;
		case PLANE: {
			const Plane *v = reinterpret_cast<const Plane *>(_data._mem);
			if (p_index == CoreStringNames::singleton->x) {
				return v->normal.x;
			} else if (p_index == CoreStringNames::singleton->y) {
				return v->normal.y;
			} else if (p_index == CoreStringNames::singleton->z) {
				return v->normal.z;
			} else if (p_index == CoreStringNames::singleton->d) {
				return v->d;
			} else if (p_index == CoreStringNames::singleton->normal) {
				return v->normal;
			}

		} break;
		case QUATERNION: {
			const Quaternion *v = reinterpret_cast<const Quaternion *>(_data._mem);
			if (p_index == CoreStringNames::singleton->x) {
				return v->x;
			} else if (p_index == CoreStringNames::singleton->y) {
				return v->y;
			} else if (p_index == CoreStringNames::singleton->z) {
				return v->z;
			} else if (p_index == CoreStringNames::singleton->w) {
				return v->w;
			}

		} break; // 10
		case AABB: {
			const ::AABB *v = _data._aabb;
			//scalar name
			if (p_index == CoreStringNames::singleton->position) {
				return v->position;
			} else if (p_index == CoreStringNames::singleton->size) {
				return v->size;
			} else if (p_index == CoreStringNames::singleton->end) {
				return v->size + v->position;
			}
		} break;
		case BASIS: {
			const Basis *v = _data._basis;
			//scalar name
			if (p_index == CoreStringNames::singleton->x) {
				return v->get_axis(0);
			} else if (p_index == CoreStringNames::singleton->y) {
				return v->get_axis(1);
			} else if (p_index == CoreStringNames::singleton->z) {
				return v->get_axis(2);
			}

		} break;
		case TRANSFORM: {
			if (p_index == CoreStringNames::singleton->basis) {
				return _data._transform->basis;
			} else if (p_index == CoreStringNames::singleton->origin) {
				return _data._transform->origin;
			}

		} break;
		case TRANSFORM2D: {
			const Transform2D *v = _data._transform2d;
			if (p_index == CoreStringNames::singleton->x) {
				return v->columns[0];
			} else if (p_index == CoreStringNames::singleton->y) {
				return v->columns[1];
			} else if (p_index == CoreStringNames::singleton->origin) {
				return v->columns[2];
			}

		} break;
		case PROJECTION: {
			const Projection *v = _data._projection;
			if (p_index == CoreStringNames::singleton->x) {
				return v->matrix[0];
			} else if (p_index == CoreStringNames::singleton->y) {
				return v->matrix[1];
			} else if (p_index == CoreStringNames::singleton->z) {
				return v->matrix[2];
			} else if (p_index == CoreStringNames::singleton->w) {
				return v->matrix[3];
			}

		} break;
		case COLOR: {
			const Color *v = reinterpret_cast<const Color *>(_data._mem);
			if (p_index == CoreStringNames::singleton->r) {
				return v->r;
			} else if (p_index == CoreStringNames::singleton->g) {
				return v->g;
			} else if (p_index == CoreStringNames::singleton->b) {
				return v->b;
			} else if (p_index == CoreStringNames::singleton->a) {
				return v->a;
			} else if (p_index == CoreStringNames::singleton->r8) {
				return int(Math::round(v->r * 255.0));
			} else if (p_index == CoreStringNames::singleton->g8) {
				return int(Math::round(v->g * 255.0));
			} else if (p_index == CoreStringNames::singleton->b8) {
				return int(Math::round(v->b * 255.0));
			} else if (p_index == CoreStringNames::singleton->a8) {
				return int(Math::round(v->a * 255.0));
			} else if (p_index == CoreStringNames::singleton->h) {
				return v->get_h();
			} else if (p_index == CoreStringNames::singleton->s) {
				return v->get_s();
			} else if (p_index == CoreStringNames::singleton->v) {
				return v->get_v();
			}
		} break;
		case OBJECT: {
			Object *obj = _OBJ_PTR(*this);
			if (unlikely(!obj)) {
				if (r_valid) {
					*r_valid = false;
				}
#ifdef DEBUG_ENABLED
				if (_get_obj().rc) {
					ERR_PRINT("Attempted get on a deleted object.");
				}
#endif
				return Variant();
			}

			return obj->get(p_index, r_valid);

		} break;
		case DICTIONARY: {
			const Dictionary *dic = reinterpret_cast<const Dictionary *>(_data._mem);
			const Variant *res = dic->getptr(p_index);
			if (!res) {
				// Backwards compatibility for before variants supported stringnames.
				const Variant *res2 = dic->getptr(p_index.operator String());

				if (res2) {
					if (r_valid) {
						*r_valid = true;
					}
					return *res2;
				}
			} else {
				if (r_valid) {
					*r_valid = true;
				}
				return *res;
			}
		} break;
		default: {
			return get(p_index.operator String(), r_valid);
		}
	}

	if (r_valid) {
		*r_valid = false;
	}
	return Variant();
}

#define DEFAULT_OP_ARRAY_CMD(m_name, m_type, skip_test, cmd)                             \
	case m_name: {                                                                       \
		skip_test;                                                                       \
                                                                                         \
		if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) { \
			int index = p_index;                                                         \
			m_type *arr = reinterpret_cast<m_type *>(_data._mem);                        \
                                                                                         \
			if (index < 0)                                                               \
				index += arr->size();                                                    \
			if (index >= 0 && index < arr->size()) {                                     \
				valid = true;                                                            \
				cmd;                                                                     \
			}                                                                            \
		}                                                                                \
	} break;

// clang-format 14 wants to add a space after the last return,
// and clang-format 15 removes it...
/* clang-format off */
#define DEFAULT_OP_DVECTOR_SET(m_name, dv_type, skip_cond) \
	DEFAULT_OP_ARRAY_CMD(m_name, PoolVector<dv_type>, if (skip_cond) return;, arr->set(index, p_value); return)
/* clang-format on */

#define DEFAULT_OP_DVECTOR_GET(m_name, dv_type) \
	DEFAULT_OP_ARRAY_CMD(m_name, const PoolVector<dv_type>, ;, return arr->get(index))

void Variant::set(const Variant &p_index, const Variant &p_value, bool *r_valid) {
	static bool _dummy = false;

	bool &valid = r_valid ? *r_valid : _dummy;
	valid = false;

	switch (type) {
		case NIL: {
			return;
		} break;
		case BOOL: {
			return;
		} break;
		case INT: {
			return;
		} break;
		case REAL: {
			return;
		} break;
		case STRING: {
			if (p_index.type != Variant::INT && p_index.type != Variant::REAL) {
				return;
			}

			int idx = p_index;
			String *str = reinterpret_cast<String *>(_data._mem);
			int len = str->length();
			if (idx < 0) {
				idx += len;
			}
			if (idx < 0 || idx >= len) {
				return;
			}

			String chr;
			if (p_value.type == Variant::INT || p_value.type == Variant::REAL) {
				chr = String::chr(p_value);
			} else if (p_value.type == Variant::STRING) {
				chr = p_value;
			} else {
				return;
			}

			*str = str->substr(0, idx) + chr + str->substr(idx + 1, len);
			valid = true;
			return;

		} break;
		case RECT2: {
			if (p_value.type == Variant::VECTOR2 || p_value.type == Variant::VECTOR2I) {
				if (p_index.get_type() == Variant::STRING) {
					//scalar name

					const String *str = reinterpret_cast<const String *>(p_index._data._mem);
					Rect2 *v = reinterpret_cast<Rect2 *>(_data._mem);
					if (*str == "position") {
						valid = true;
						v->position = p_value;
						return;
					} else if (*str == "size") {
						valid = true;
						v->size = p_value;
						return;
					} else if (*str == "end") {
						valid = true;
						v->size = Vector2(p_value) - v->position;
						return;
					}
				} else if (p_index.get_type() == Variant::STRING_NAME) {
					//scalar name

					Rect2 *v = reinterpret_cast<Rect2 *>(_data._mem);
					if (p_index == CoreStringNames::singleton->position) {
						valid = true;
						v->position = p_value;
						return;
					} else if (p_index == CoreStringNames::singleton->size) {
						valid = true;
						v->size = p_value;
						return;
					} else if (p_index == CoreStringNames::singleton->end) {
						valid = true;
						v->size = Vector2(p_value) - v->position;
						return;
					}
				}
			} else {
				return;
			}
		} break; //7
		case RECT2I: {
			if (p_value.type == Variant::VECTOR2 || p_value.type == Variant::VECTOR2I) {
				if (p_index.get_type() == Variant::STRING) {
					//scalar name

					const String *str = reinterpret_cast<const String *>(p_index._data._mem);
					Rect2i *v = reinterpret_cast<Rect2i *>(_data._mem);
					if (*str == "position") {
						valid = true;
						v->position = p_value;
						return;
					} else if (*str == "size") {
						valid = true;
						v->size = p_value;
						return;
					} else if (*str == "end") {
						valid = true;
						//TODO fix
						v->size = Vector2i(Vector2(p_value)) - v->position;
						return;
					}
				} else if (p_index.get_type() == Variant::STRING_NAME) {
					//scalar name

					Rect2i *v = reinterpret_cast<Rect2i *>(_data._mem);
					if (p_index == CoreStringNames::singleton->position) {
						valid = true;
						v->position = p_value;
						return;
					} else if (p_index == CoreStringNames::singleton->size) {
						valid = true;
						v->size = p_value;
						return;
					} else if (p_index == CoreStringNames::singleton->end) {
						valid = true;
						v->size = Vector2(p_value) - v->position;
						return;
					}
				}
			} else {
				return;
			}
		} break;
		case VECTOR2: {
			if (p_value.type != Variant::INT && p_value.type != Variant::REAL) {
				return;
			}

			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				// scalar index
				int idx = p_index;

				if (idx < 0) {
					idx += 2;
				}
				if (idx >= 0 && idx < 2) {
					Vector2 *v = reinterpret_cast<Vector2 *>(_data._mem);
					valid = true;
					(*v)[idx] = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING) {
				//scalar name

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Vector2 *v = reinterpret_cast<Vector2 *>(_data._mem);
				if (*str == "x") {
					valid = true;
					v->x = p_value;
					return;
				} else if (*str == "y") {
					valid = true;
					v->y = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				//scalar name

				Vector2 *v = reinterpret_cast<Vector2 *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					valid = true;
					v->x = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->y) {
					valid = true;
					v->y = p_value;
					return;
				}
			}
		} break; // 5
		case VECTOR2I: {
			if (p_value.type != Variant::INT && p_value.type != Variant::REAL) {
				return;
			}

			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				// scalar index
				int idx = p_index;

				if (idx < 0) {
					idx += 2;
				}
				if (idx >= 0 && idx < 2) {
					Vector2i *v = reinterpret_cast<Vector2i *>(_data._mem);
					valid = true;
					(*v)[idx] = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING) {
				//scalar name

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Vector2i *v = reinterpret_cast<Vector2i *>(_data._mem);
				if (*str == "x") {
					valid = true;
					v->x = p_value;
					return;
				} else if (*str == "y") {
					valid = true;
					v->y = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				//scalar name

				Vector2i *v = reinterpret_cast<Vector2i *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					valid = true;
					v->x = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->y) {
					valid = true;
					v->y = p_value;
					return;
				}
			}

		} break; //6

		case VECTOR3: {
			if (p_value.type != Variant::INT && p_value.type != Variant::REAL) {
				return;
			}

			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				//scalar index
				int idx = p_index;
				if (idx < 0) {
					idx += 3;
				}
				if (idx >= 0 && idx < 3) {
					Vector3 *v = reinterpret_cast<Vector3 *>(_data._mem);
					valid = true;
					(*v)[idx] = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING) {
				//scalar name
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Vector3 *v = reinterpret_cast<Vector3 *>(_data._mem);
				if (*str == "x") {
					valid = true;
					v->x = p_value;
					return;
				} else if (*str == "y") {
					valid = true;
					v->y = p_value;
					return;
				} else if (*str == "z") {
					valid = true;
					v->z = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				//scalar name

				Vector3 *v = reinterpret_cast<Vector3 *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					valid = true;
					v->x = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->y) {
					valid = true;
					v->y = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->z) {
					valid = true;
					v->z = p_value;
					return;
				}
			}

		} break;
		case VECTOR3I: {
			if (p_value.type != Variant::INT && p_value.type != Variant::REAL) {
				return;
			}

			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				//scalar index
				int idx = p_index;
				if (idx < 0) {
					idx += 3;
				}
				if (idx >= 0 && idx < 3) {
					Vector3i *v = reinterpret_cast<Vector3i *>(_data._mem);
					valid = true;
					(*v)[idx] = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING) {
				//scalar name
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Vector3 *v = reinterpret_cast<Vector3 *>(_data._mem);
				if (*str == "x") {
					valid = true;
					v->x = p_value;
					return;
				} else if (*str == "y") {
					valid = true;
					v->y = p_value;
					return;
				} else if (*str == "z") {
					valid = true;
					v->z = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				//scalar name

				Vector3i *v = reinterpret_cast<Vector3i *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					valid = true;
					v->x = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->y) {
					valid = true;
					v->y = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->z) {
					valid = true;
					v->z = p_value;
					return;
				}
			}

		} break;
		case VECTOR4: {
			if (p_value.type != Variant::INT && p_value.type != Variant::REAL) {
				return;
			}

			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				//scalar index
				int idx = p_index;
				if (idx < 0) {
					idx += 4;
				}
				if (idx >= 0 && idx < 4) {
					Vector4 *v = reinterpret_cast<Vector4 *>(_data._mem);
					valid = true;
					(*v)[idx] = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING) {
				//scalar name
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Vector4 *v = reinterpret_cast<Vector4 *>(_data._mem);
				if (*str == "x") {
					valid = true;
					v->x = p_value;
					return;
				} else if (*str == "y") {
					valid = true;
					v->y = p_value;
					return;
				} else if (*str == "z") {
					valid = true;
					v->z = p_value;
					return;
				} else if (*str == "w") {
					valid = true;
					v->w = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				//scalar name

				Vector4 *v = reinterpret_cast<Vector4 *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					valid = true;
					v->x = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->y) {
					valid = true;
					v->y = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->z) {
					valid = true;
					v->z = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->w) {
					valid = true;
					v->w = p_value;
					return;
				}
			}
		} break;
		case VECTOR4I: {
			if (p_value.type != Variant::INT && p_value.type != Variant::REAL) {
				return;
			}

			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				//scalar index
				int idx = p_index;
				if (idx < 0) {
					idx += 4;
				}
				if (idx >= 0 && idx < 4) {
					Vector4i *v = reinterpret_cast<Vector4i *>(_data._mem);
					valid = true;
					(*v)[idx] = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING) {
				//scalar name
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Vector4 *v = reinterpret_cast<Vector4 *>(_data._mem);
				if (*str == "x") {
					valid = true;
					v->x = p_value;
					return;
				} else if (*str == "y") {
					valid = true;
					v->y = p_value;
					return;
				} else if (*str == "z") {
					valid = true;
					v->z = p_value;
					return;
				} else if (*str == "w") {
					valid = true;
					v->w = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				//scalar name

				Vector4i *v = reinterpret_cast<Vector4i *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					valid = true;
					v->x = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->y) {
					valid = true;
					v->y = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->z) {
					valid = true;
					v->z = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->w) {
					valid = true;
					v->w = p_value;
					return;
				}
			}

		} break;
		case PLANE: {
			if (p_index.get_type() == Variant::STRING) {
				//scalar name
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Plane *v = reinterpret_cast<Plane *>(_data._mem);
				if (*str == "x") {
					if (p_value.type != Variant::INT && p_value.type != Variant::REAL) {
						return;
					}

					valid = true;
					v->normal.x = p_value;
					return;
				} else if (*str == "y") {
					if (p_value.type != Variant::INT && p_value.type != Variant::REAL) {
						return;
					}

					valid = true;
					v->normal.y = p_value;
					return;
				} else if (*str == "z") {
					if (p_value.type != Variant::INT && p_value.type != Variant::REAL) {
						return;
					}

					valid = true;
					v->normal.z = p_value;
					return;
				} else if (*str == "normal") {
					if (p_value.type != Variant::VECTOR3) {
						return;
					}

					valid = true;
					v->normal = p_value;
					return;
				} else if (*str == "d") {
					valid = true;
					v->d = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				//scalar name

				Plane *v = reinterpret_cast<Plane *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					if (p_value.type != Variant::INT && p_value.type != Variant::REAL) {
						return;
					}

					valid = true;
					v->normal.x = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->y) {
					if (p_value.type != Variant::INT && p_value.type != Variant::REAL) {
						return;
					}

					valid = true;
					v->normal.y = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->z) {
					if (p_value.type != Variant::INT && p_value.type != Variant::REAL) {
						return;
					}

					valid = true;
					v->normal.z = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->normal) {
					if (p_value.type != Variant::VECTOR3) {
						return;
					}

					valid = true;
					v->normal = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->d) {
					valid = true;
					v->d = p_value;
					return;
				}
			}

		} break;
		case QUATERNION: {
			if (p_value.type != Variant::INT && p_value.type != Variant::REAL) {
				return;
			}

			if (p_index.get_type() == Variant::STRING) {
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Quaternion *v = reinterpret_cast<Quaternion *>(_data._mem);
				if (*str == "x") {
					valid = true;
					v->x = p_value;
					return;
				} else if (*str == "y") {
					valid = true;
					v->y = p_value;
					return;
				} else if (*str == "z") {
					valid = true;
					v->z = p_value;
					return;
				} else if (*str == "w") {
					valid = true;
					v->w = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				Quaternion *v = reinterpret_cast<Quaternion *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					valid = true;
					v->x = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->y) {
					valid = true;
					v->y = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->z) {
					valid = true;
					v->z = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->w) {
					valid = true;
					v->w = p_value;
					return;
				}
			}

		} break; // 10
		case AABB: {
			if (p_value.type != Variant::VECTOR3) {
				return;
			}

			if (p_index.get_type() == Variant::STRING) {
				//scalar name

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				::AABB *v = _data._aabb;
				if (*str == "position") {
					valid = true;
					v->position = p_value;
					return;
				} else if (*str == "size") {
					valid = true;
					v->size = p_value;
					return;
				} else if (*str == "end") {
					valid = true;
					v->size = Vector3(p_value) - v->position;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				//scalar name

				::AABB *v = _data._aabb;
				if (p_index == CoreStringNames::singleton->position) {
					valid = true;
					v->position = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->size) {
					valid = true;
					v->size = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->end) {
					valid = true;
					v->size = Vector3(p_value) - v->position;
					return;
				}
			}
		} break;
		case BASIS: {
			if (p_value.type != Variant::VECTOR3) {
				return;
			}

			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				int index = p_index;

				if (index < 0) {
					index += 3;
				}
				if (index >= 0 && index < 3) {
					Basis *v = _data._basis;

					valid = true;
					v->set_axis(index, p_value);
					return;
				}
			} else if (p_index.get_type() == Variant::STRING) {
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Basis *v = _data._basis;

				if (*str == "x") {
					valid = true;
					v->set_axis(0, p_value);
					return;
				} else if (*str == "y") {
					valid = true;
					v->set_axis(1, p_value);
					return;
				} else if (*str == "z") {
					valid = true;
					v->set_axis(2, p_value);
					return;
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				Basis *v = _data._basis;

				if (p_index == CoreStringNames::singleton->x) {
					valid = true;
					v->set_axis(0, p_value);
					return;
				} else if (p_index == CoreStringNames::singleton->y) {
					valid = true;
					v->set_axis(1, p_value);
					return;
				} else if (p_index == CoreStringNames::singleton->z) {
					valid = true;
					v->set_axis(2, p_value);
					return;
				}
			}

		} break;
		case TRANSFORM: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				if (p_value.type != Variant::VECTOR3) {
					return;
				}

				int index = p_index;

				if (index < 0) {
					index += 4;
				}
				if (index >= 0 && index < 4) {
					Transform *v = _data._transform;
					valid = true;
					if (index == 3) {
						v->origin = p_value;
					} else {
						v->basis.set_axis(index, p_value);
					}
					return;
				}
			} else if (p_index.get_type() == Variant::STRING) {
				Transform *v = _data._transform;
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);

				if (*str == "basis") {
					if (p_value.type != Variant::BASIS) {
						return;
					}
					valid = true;
					v->basis = p_value;
					return;
				}
				if (*str == "origin") {
					if (p_value.type != Variant::VECTOR3) {
						return;
					}
					valid = true;
					v->origin = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				Transform *v = _data._transform;

				if (p_index == CoreStringNames::singleton->basis) {
					if (p_value.type != Variant::BASIS) {
						return;
					}
					valid = true;
					v->basis = p_value;
					return;
				}
				if (p_index == CoreStringNames::singleton->origin) {
					if (p_value.type != Variant::VECTOR3) {
						return;
					}
					valid = true;
					v->origin = p_value;
					return;
				}
			}

		} break;
		case TRANSFORM2D: {
			if (p_value.type != Variant::VECTOR2 || p_value.get_type() != Variant::VECTOR2I) {
				return;
			}

			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				int index = p_index;

				if (index < 0) {
					index += 3;
				}
				if (index >= 0 && index < 3) {
					Transform2D *v = _data._transform2d;

					valid = true;
					v->columns[index] = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING) {
				//scalar name
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Transform2D *v = _data._transform2d;
				if (*str == "x") {
					valid = true;
					v->columns[0] = p_value;
					return;
				} else if (*str == "y") {
					valid = true;
					v->columns[1] = p_value;
					return;
				} else if (*str == "origin") {
					valid = true;
					v->columns[2] = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				//scalar name

				Transform2D *v = _data._transform2d;
				if (p_index == CoreStringNames::singleton->x) {
					valid = true;
					v->columns[0] = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->y) {
					valid = true;
					v->columns[1] = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->origin) {
					valid = true;
					v->columns[2] = p_value;
					return;
				}
			}

		} break;
		case PROJECTION: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				if (p_value.type != Variant::VECTOR4) {
					return;
				}

				int index = p_index;

				if (index < 0) {
					index += 4;
				}
				if (index >= 0 && index < 4) {
					Projection *v = _data._projection;
					valid = true;
					v->matrix[index] = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING) {
				Projection *v = _data._projection;
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);

				if (p_value.type != Variant::VECTOR4) {
					return;
				}

				if (*str == "x") {
					valid = true;
					v->matrix[0] = p_value;
					return;
				} else if (*str == "y") {
					valid = true;
					v->matrix[1] = p_value;
					return;
				} else if (*str == "z") {
					valid = true;
					v->matrix[2] = p_value;
					return;
				} else if (*str == "w") {
					valid = true;
					v->matrix[3] = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				Projection *v = _data._projection;

				if (p_value.type != Variant::VECTOR4) {
					return;
				}

				if (p_index == CoreStringNames::singleton->x) {
					valid = true;
					v->matrix[0] = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->y) {
					valid = true;
					v->matrix[1] = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->z) {
					valid = true;
					v->matrix[2] = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->w) {
					valid = true;
					v->matrix[3] = p_value;
					return;
				}
			}

		} break;
		case COLOR: {
			if (p_value.type != Variant::INT && p_value.type != Variant::REAL) {
				return;
			}

			if (p_index.get_type() == Variant::STRING) {
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Color *v = reinterpret_cast<Color *>(_data._mem);
				if (*str == "r") {
					valid = true;
					v->r = p_value;
					return;
				} else if (*str == "g") {
					valid = true;
					v->g = p_value;
					return;
				} else if (*str == "b") {
					valid = true;
					v->b = p_value;
					return;
				} else if (*str == "a") {
					valid = true;
					v->a = p_value;
					return;
				} else if (*str == "h") {
					valid = true;
					v->set_hsv(p_value, v->get_s(), v->get_v(), v->a);
					return;
				} else if (*str == "s") {
					valid = true;
					v->set_hsv(v->get_h(), p_value, v->get_v(), v->a);
					return;
				} else if (*str == "v") {
					valid = true;
					v->set_hsv(v->get_h(), v->get_s(), p_value, v->a);
					return;
				} else if (*str == "r8") {
					valid = true;
					v->r = float(p_value) / 255.0;
					return;
				} else if (*str == "g8") {
					valid = true;
					v->g = float(p_value) / 255.0;
					return;
				} else if (*str == "b8") {
					valid = true;
					v->b = float(p_value) / 255.0;
					return;
				} else if (*str == "a8") {
					valid = true;
					v->a = float(p_value) / 255.0;
					return;
				}
			} else if (p_index.get_type() == Variant::INT) {
				int idx = p_index;
				if (idx < 0) {
					idx += 4;
				}
				if (idx >= 0 && idx < 4) {
					Color *v = reinterpret_cast<Color *>(_data._mem);
					(*v)[idx] = p_value;
					valid = true;
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				Color *v = reinterpret_cast<Color *>(_data._mem);
				if (p_index == CoreStringNames::singleton->r) {
					valid = true;
					v->r = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->g) {
					valid = true;
					v->g = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->b) {
					valid = true;
					v->b = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->a) {
					valid = true;
					v->a = p_value;
					return;
				} else if (p_index == CoreStringNames::singleton->h) {
					valid = true;
					v->set_hsv(p_value, v->get_s(), v->get_v(), v->a);
					return;
				} else if (p_index == CoreStringNames::singleton->s) {
					valid = true;
					v->set_hsv(v->get_h(), p_value, v->get_v(), v->a);
					return;
				} else if (p_index == CoreStringNames::singleton->v) {
					valid = true;
					v->set_hsv(v->get_h(), v->get_s(), p_value, v->a);
					return;
				} else if (p_index == CoreStringNames::singleton->r8) {
					valid = true;
					v->r = float(p_value) / 255.0;
					return;
				} else if (p_index == CoreStringNames::singleton->g8) {
					valid = true;
					v->g = float(p_value) / 255.0;
					return;
				} else if (p_index == CoreStringNames::singleton->b8) {
					valid = true;
					v->b = float(p_value) / 255.0;
					return;
				} else if (p_index == CoreStringNames::singleton->a8) {
					valid = true;
					v->a = float(p_value) / 255.0;
					return;
				}
			}

		} break;
		case OBJECT: {
			Object *obj = _OBJ_PTR(*this);
			if (unlikely(!obj)) {
				valid = false;
#ifdef DEBUG_ENABLED
				if (_get_obj().rc) {
					ERR_PRINT("Attempted set on a deleted object.");
				}
#endif
				return;
			}

			obj->set(p_index, p_value, r_valid);
			return;
		} break;
		case STRING_NAME: {
		} break;
		case DICTIONARY: {
			Dictionary *dic = reinterpret_cast<Dictionary *>(_data._mem);
			dic->operator[](p_index) = p_value;
			valid = true; //always valid, i guess? should this really be ok?
			return;
		} break;
			// clang-format 14 wants to add a space after the last return,
			// and clang-format 15 removes it...
			/* clang-format off */
			DEFAULT_OP_ARRAY_CMD(ARRAY, Array, ;, (*arr)[index] = p_value; return) // 20
			/* clang-format on */
			DEFAULT_OP_DVECTOR_SET(POOL_BYTE_ARRAY, uint8_t, p_value.type != Variant::REAL && p_value.type != Variant::INT)
			DEFAULT_OP_DVECTOR_SET(POOL_INT_ARRAY, int, p_value.type != Variant::REAL && p_value.type != Variant::INT)
			DEFAULT_OP_DVECTOR_SET(POOL_REAL_ARRAY, real_t, p_value.type != Variant::REAL && p_value.type != Variant::INT)
			DEFAULT_OP_DVECTOR_SET(POOL_STRING_ARRAY, String, p_value.type != Variant::STRING)
			DEFAULT_OP_DVECTOR_SET(POOL_VECTOR2_ARRAY, Vector2, p_value.type != Variant::VECTOR2) // 25
			DEFAULT_OP_DVECTOR_SET(POOL_VECTOR2I_ARRAY, Vector2i, p_value.type != Variant::VECTOR2I)
			DEFAULT_OP_DVECTOR_SET(POOL_VECTOR3_ARRAY, Vector3, p_value.type != Variant::VECTOR3)
			DEFAULT_OP_DVECTOR_SET(POOL_VECTOR3I_ARRAY, Vector3i, p_value.type != Variant::VECTOR3I)
			DEFAULT_OP_DVECTOR_SET(POOL_VECTOR4_ARRAY, Vector4, p_value.type != Variant::VECTOR4)
			DEFAULT_OP_DVECTOR_SET(POOL_VECTOR4I_ARRAY, Vector4i, p_value.type != Variant::VECTOR4I)
			DEFAULT_OP_DVECTOR_SET(POOL_COLOR_ARRAY, Color, p_value.type != Variant::COLOR)
		default:
			return;
	}
}

Variant Variant::get(const Variant &p_index, bool *r_valid) const {
	static bool _dummy = false;

	bool &valid = r_valid ? *r_valid : _dummy;

	valid = false;

	switch (type) {
		case NIL: {
			return Variant();
		} break;
		case BOOL: {
			return Variant();
		} break;
		case INT: {
			return Variant();
		} break;
		case REAL: {
			return Variant();
		} break;
		case STRING: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				//string index

				int idx = p_index;
				const String *str = reinterpret_cast<const String *>(_data._mem);
				if (idx < 0) {
					idx += str->length();
				}
				if (idx >= 0 && idx < str->length()) {
					valid = true;
					return str->substr(idx, 1);
				}
			}

		} break;
		case RECT2: {
			if (p_index.get_type() == Variant::STRING) {
				//scalar name

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Rect2 *v = reinterpret_cast<const Rect2 *>(_data._mem);
				if (*str == "position") {
					valid = true;
					return v->position;
				} else if (*str == "size") {
					valid = true;
					return v->size;
				} else if (*str == "end") {
					valid = true;
					return v->size + v->position;
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				//scalar name

				const Rect2 *v = reinterpret_cast<const Rect2 *>(_data._mem);
				if (p_index == CoreStringNames::singleton->position) {
					valid = true;
					return v->position;
				} else if (p_index == CoreStringNames::singleton->size) {
					valid = true;
					return v->size;
				} else if (p_index == CoreStringNames::singleton->end) {
					valid = true;
					return v->size + v->position;
				}
			}
		} break;
		case RECT2I: {
			if (p_index.get_type() == Variant::STRING) {
				//scalar name

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Rect2i *v = reinterpret_cast<const Rect2i *>(_data._mem);
				if (*str == "position") {
					valid = true;
					return v->position;
				} else if (*str == "size") {
					valid = true;
					return v->size;
				} else if (*str == "end") {
					valid = true;
					return v->size + v->position;
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				//scalar name

				const Rect2i *v = reinterpret_cast<const Rect2i *>(_data._mem);
				if (p_index == CoreStringNames::singleton->position) {
					valid = true;
					return v->position;
				} else if (p_index == CoreStringNames::singleton->size) {
					valid = true;
					return v->size;
				} else if (p_index == CoreStringNames::singleton->end) {
					valid = true;
					return v->size + v->position;
				}
			}
		} break;
		case VECTOR2: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				// scalar index
				int idx = p_index;
				if (idx < 0) {
					idx += 2;
				}
				if (idx >= 0 && idx < 2) {
					const Vector2 *v = reinterpret_cast<const Vector2 *>(_data._mem);
					valid = true;
					return (*v)[idx];
				}
			} else if (p_index.get_type() == Variant::STRING) {
				//scalar name

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Vector2 *v = reinterpret_cast<const Vector2 *>(_data._mem);
				if (*str == "x") {
					valid = true;
					return v->x;
				} else if (*str == "y") {
					valid = true;
					return v->y;
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				//scalar name

				const Vector2 *v = reinterpret_cast<const Vector2 *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					valid = true;
					return v->x;
				} else if (p_index == CoreStringNames::singleton->y) {
					valid = true;
					return v->y;
				}
			}

		} break; // 5
		case VECTOR2I: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				// scalar index
				int idx = p_index;
				if (idx < 0) {
					idx += 2;
				}
				if (idx >= 0 && idx < 2) {
					const Vector2i *v = reinterpret_cast<const Vector2i *>(_data._mem);
					valid = true;
					return (*v)[idx];
				}
			} else if (p_index.get_type() == Variant::STRING) {
				//scalar name

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Vector2i *v = reinterpret_cast<const Vector2i *>(_data._mem);
				if (*str == "x") {
					valid = true;
					return v->x;
				} else if (*str == "y") {
					valid = true;
					return v->y;
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				//scalar name

				const Vector2i *v = reinterpret_cast<const Vector2i *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					valid = true;
					return v->x;
				} else if (p_index == CoreStringNames::singleton->y) {
					valid = true;
					return v->y;
				}
			}

		} break; // 6
		case VECTOR3: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				//scalar index
				int idx = p_index;
				if (idx < 0) {
					idx += 3;
				}
				if (idx >= 0 && idx < 3) {
					const Vector3 *v = reinterpret_cast<const Vector3 *>(_data._mem);
					valid = true;
					return (*v)[idx];
				}
			} else if (p_index.get_type() == Variant::STRING) {
				//scalar name
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Vector3 *v = reinterpret_cast<const Vector3 *>(_data._mem);
				if (*str == "x") {
					valid = true;
					return v->x;
				} else if (*str == "y") {
					valid = true;
					return v->y;
				} else if (*str == "z") {
					valid = true;
					return v->z;
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				//scalar name

				const Vector3 *v = reinterpret_cast<const Vector3 *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					valid = true;
					return v->x;
				} else if (p_index == CoreStringNames::singleton->y) {
					valid = true;
					return v->y;
				} else if (p_index == CoreStringNames::singleton->z) {
					valid = true;
					return v->z;
				}
			}

		} break;
		case VECTOR3I: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				//scalar index
				int idx = p_index;
				if (idx < 0) {
					idx += 3;
				}
				if (idx >= 0 && idx < 3) {
					const Vector3i *v = reinterpret_cast<const Vector3i *>(_data._mem);
					valid = true;
					return (*v)[idx];
				}
			} else if (p_index.get_type() == Variant::STRING) {
				//scalar name
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Vector3i *v = reinterpret_cast<const Vector3i *>(_data._mem);
				if (*str == "x") {
					valid = true;
					return v->x;
				} else if (*str == "y") {
					valid = true;
					return v->y;
				} else if (*str == "z") {
					valid = true;
					return v->z;
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				//scalar name

				const Vector3i *v = reinterpret_cast<const Vector3i *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					valid = true;
					return v->x;
				} else if (p_index == CoreStringNames::singleton->y) {
					valid = true;
					return v->y;
				} else if (p_index == CoreStringNames::singleton->z) {
					valid = true;
					return v->z;
				}
			}

		} break;
		case VECTOR4: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				//scalar index
				int idx = p_index;
				if (idx < 0) {
					idx += 4;
				}
				if (idx >= 0 && idx < 4) {
					const Vector4 *v = reinterpret_cast<const Vector4 *>(_data._mem);
					valid = true;
					return (*v)[idx];
				}
			} else if (p_index.get_type() == Variant::STRING) {
				//scalar name
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Vector4 *v = reinterpret_cast<const Vector4 *>(_data._mem);
				if (*str == "x") {
					valid = true;
					return v->x;
				} else if (*str == "y") {
					valid = true;
					return v->y;
				} else if (*str == "z") {
					valid = true;
					return v->z;
				} else if (*str == "w") {
					valid = true;
					return v->w;
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				//scalar name

				const Vector4 *v = reinterpret_cast<const Vector4 *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					valid = true;
					return v->x;
				} else if (p_index == CoreStringNames::singleton->y) {
					valid = true;
					return v->y;
				} else if (p_index == CoreStringNames::singleton->z) {
					valid = true;
					return v->z;
				} else if (p_index == CoreStringNames::singleton->w) {
					valid = true;
					return v->w;
				}
			}

		} break;
		case VECTOR4I: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				//scalar index
				int idx = p_index;
				if (idx < 0) {
					idx += 4;
				}
				if (idx >= 0 && idx < 4) {
					const Vector4i *v = reinterpret_cast<const Vector4i *>(_data._mem);
					valid = true;
					return (*v)[idx];
				}
			} else if (p_index.get_type() == Variant::STRING) {
				//scalar name
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Vector4i *v = reinterpret_cast<const Vector4i *>(_data._mem);
				if (*str == "x") {
					valid = true;
					return v->x;
				} else if (*str == "y") {
					valid = true;
					return v->y;
				} else if (*str == "z") {
					valid = true;
					return v->z;
				} else if (*str == "w") {
					valid = true;
					return v->w;
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				//scalar name
				const Vector4i *v = reinterpret_cast<const Vector4i *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					valid = true;
					return v->x;
				} else if (p_index == CoreStringNames::singleton->y) {
					valid = true;
					return v->y;
				} else if (p_index == CoreStringNames::singleton->z) {
					valid = true;
					return v->z;
				} else if (p_index == CoreStringNames::singleton->w) {
					valid = true;
					return v->w;
				}
			}

		} break;
		case PLANE: {
			if (p_index.get_type() == Variant::STRING) {
				//scalar name
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Plane *v = reinterpret_cast<const Plane *>(_data._mem);
				if (*str == "x") {
					valid = true;
					return v->normal.x;
				} else if (*str == "y") {
					valid = true;
					return v->normal.y;
				} else if (*str == "z") {
					valid = true;
					return v->normal.z;
				} else if (*str == "normal") {
					valid = true;
					return v->normal;
				} else if (*str == "d") {
					valid = true;
					return v->d;
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				//scalar name

				const Plane *v = reinterpret_cast<const Plane *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					valid = true;
					return v->normal.x;
				} else if (p_index == CoreStringNames::singleton->y) {
					valid = true;
					return v->normal.y;
				} else if (p_index == CoreStringNames::singleton->z) {
					valid = true;
					return v->normal.z;
				} else if (p_index == CoreStringNames::singleton->normal) {
					valid = true;
					return v->normal;
				} else if (p_index == CoreStringNames::singleton->d) {
					valid = true;
					return v->d;
				}
			}

		} break;
		case QUATERNION: {
			if (p_index.get_type() == Variant::STRING) {
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Quaternion *v = reinterpret_cast<const Quaternion *>(_data._mem);
				if (*str == "x") {
					valid = true;
					return v->x;
				} else if (*str == "y") {
					valid = true;
					return v->y;
				} else if (*str == "z") {
					valid = true;
					return v->z;
				} else if (*str == "w") {
					valid = true;
					return v->w;
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				const Quaternion *v = reinterpret_cast<const Quaternion *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					valid = true;
					return v->x;
				} else if (p_index == CoreStringNames::singleton->y) {
					valid = true;
					return v->y;
				} else if (p_index == CoreStringNames::singleton->z) {
					valid = true;
					return v->z;
				} else if (p_index == CoreStringNames::singleton->w) {
					valid = true;
					return v->w;
				}
			}

		} break; // 10
		case AABB: {
			if (p_index.get_type() == Variant::STRING) {
				//scalar name

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const ::AABB *v = _data._aabb;
				if (*str == "position") {
					valid = true;
					return v->position;
				} else if (*str == "size") {
					valid = true;
					return v->size;
				} else if (*str == "end") {
					valid = true;
					return v->size + v->position;
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				//scalar name

				const ::AABB *v = _data._aabb;
				if (p_index == CoreStringNames::singleton->position) {
					valid = true;
					return v->position;
				} else if (p_index == CoreStringNames::singleton->size) {
					valid = true;
					return v->size;
				} else if (p_index == CoreStringNames::singleton->end) {
					valid = true;
					return v->size + v->position;
				}
			}
		} break;
		case BASIS: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				int index = p_index;
				if (index < 0) {
					index += 3;
				}
				if (index >= 0 && index < 3) {
					const Basis *v = _data._basis;

					valid = true;
					return v->get_axis(index);
				}
			} else if (p_index.get_type() == Variant::STRING) {
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Basis *v = _data._basis;

				if (*str == "x") {
					valid = true;
					return v->get_axis(0);
				} else if (*str == "y") {
					valid = true;
					return v->get_axis(1);
				} else if (*str == "z") {
					valid = true;
					return v->get_axis(2);
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				const Basis *v = _data._basis;

				if (p_index == CoreStringNames::singleton->x) {
					valid = true;
					return v->get_axis(0);
				} else if (p_index == CoreStringNames::singleton->y) {
					valid = true;
					return v->get_axis(1);
				} else if (p_index == CoreStringNames::singleton->z) {
					valid = true;
					return v->get_axis(2);
				}
			}

		} break;
		case TRANSFORM: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				int index = p_index;
				if (index < 0) {
					index += 4;
				}
				if (index >= 0 && index < 4) {
					const Transform *v = _data._transform;
					valid = true;
					return index == 3 ? v->origin : v->basis.get_axis(index);
				}
			} else if (p_index.get_type() == Variant::STRING) {
				const Transform *v = _data._transform;
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);

				if (*str == "basis") {
					valid = true;
					return v->basis;
				}
				if (*str == "origin") {
					valid = true;
					return v->origin;
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				const Transform *v = _data._transform;

				if (p_index == CoreStringNames::singleton->basis) {
					valid = true;
					return v->basis;
				}
				if (p_index == CoreStringNames::singleton->origin) {
					valid = true;
					return v->origin;
				}
			}

		} break;
		case TRANSFORM2D: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				int index = p_index;

				if (index < 0) {
					index += 3;
				}
				if (index >= 0 && index < 3) {
					const Transform2D *v = _data._transform2d;

					valid = true;
					return v->columns[index];
				}
			} else if (p_index.get_type() == Variant::STRING) {
				//scalar name
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Transform2D *v = _data._transform2d;
				if (*str == "x") {
					valid = true;
					return v->columns[0];
				} else if (*str == "y") {
					valid = true;
					return v->columns[1];
				} else if (*str == "origin") {
					valid = true;
					return v->columns[2];
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				//scalar name

				const Transform2D *v = _data._transform2d;
				if (p_index == CoreStringNames::singleton->x) {
					valid = true;
					return v->columns[0];
				} else if (p_index == CoreStringNames::singleton->y) {
					valid = true;
					return v->columns[1];
				} else if (p_index == CoreStringNames::singleton->origin) {
					valid = true;
					return v->columns[2];
				}
			}

		} break;
		case PROJECTION: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				int index = p_index;
				if (index < 0) {
					index += 4;
				}
				if (index >= 0 && index < 4) {
					const Projection *v = _data._projection;
					valid = true;
					return v->matrix[index];
				}
			} else if (p_index.get_type() == Variant::STRING) {
				const Projection *v = _data._projection;
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);

				if (*str == "x") {
					valid = true;
					return v->matrix[0];
				} else if (*str == "y") {
					valid = true;
					return v->matrix[1];
				} else if (*str == "z") {
					valid = true;
					return v->matrix[2];
				} else if (*str == "w") {
					valid = true;
					return v->matrix[3];
				}
			} else if (p_index.get_type() == Variant::STRING_NAME) {
				const Projection *v = _data._projection;

				if (p_index == CoreStringNames::singleton->x) {
					valid = true;
					return v->matrix[0];
				} else if (p_index == CoreStringNames::singleton->y) {
					valid = true;
					return v->matrix[1];
				} else if (p_index == CoreStringNames::singleton->z) {
					valid = true;
					return v->matrix[2];
				} else if (p_index == CoreStringNames::singleton->w) {
					valid = true;
					return v->matrix[3];
				}
			}

		} break;
		case COLOR: {
			if (p_index.get_type() == Variant::STRING) {
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Color *v = reinterpret_cast<const Color *>(_data._mem);
				if (*str == "r") {
					valid = true;
					return v->r;
				} else if (*str == "g") {
					valid = true;
					return v->g;
				} else if (*str == "b") {
					valid = true;
					return v->b;
				} else if (*str == "a") {
					valid = true;
					return v->a;
				} else if (*str == "h") {
					valid = true;
					return v->get_h();
				} else if (*str == "s") {
					valid = true;
					return v->get_s();
				} else if (*str == "v") {
					valid = true;
					return v->get_v();
				} else if (*str == "r8") {
					valid = true;
					return (int)Math::round(v->r * 255.0);
				} else if (*str == "g8") {
					valid = true;
					return (int)Math::round(v->g * 255.0);
				} else if (*str == "b8") {
					valid = true;
					return (int)Math::round(v->b * 255.0);
				} else if (*str == "a8") {
					valid = true;
					return (int)Math::round(v->a * 255.0);
				}
			} else if (p_index.get_type() == Variant::INT) {
				int idx = p_index;
				if (idx < 0) {
					idx += 4;
				}
				if (idx >= 0 && idx < 4) {
					const Color *v = reinterpret_cast<const Color *>(_data._mem);
					valid = true;
					return (*v)[idx];
				}
			} else if (p_index.get_type() == Variant::STRING) {
				const Color *v = reinterpret_cast<const Color *>(_data._mem);
				if (p_index == CoreStringNames::singleton->r) {
					valid = true;
					return v->r;
				} else if (p_index == CoreStringNames::singleton->g) {
					valid = true;
					return v->g;
				} else if (p_index == CoreStringNames::singleton->b) {
					valid = true;
					return v->b;
				} else if (p_index == CoreStringNames::singleton->a) {
					valid = true;
					return v->a;
				} else if (p_index == CoreStringNames::singleton->h) {
					valid = true;
					return v->get_h();
				} else if (p_index == CoreStringNames::singleton->s) {
					valid = true;
					return v->get_s();
				} else if (p_index == CoreStringNames::singleton->v) {
					valid = true;
					return v->get_v();
				} else if (p_index == CoreStringNames::singleton->r8) {
					valid = true;
					return (int)Math::round(v->r * 255.0);
				} else if (p_index == CoreStringNames::singleton->g8) {
					valid = true;
					return (int)Math::round(v->g * 255.0);
				} else if (p_index == CoreStringNames::singleton->b8) {
					valid = true;
					return (int)Math::round(v->b * 255.0);
				} else if (p_index == CoreStringNames::singleton->a8) {
					valid = true;
					return (int)Math::round(v->a * 255.0);
				}
			}

		} break;
		case OBJECT: {
			Object *obj = _OBJ_PTR(*this);
			if (unlikely(!obj)) {
				valid = false;
#ifdef DEBUG_ENABLED
				if (_get_obj().rc) {
					ERR_PRINT("Attempted get on a deleted object.");
				}
#endif
				return Variant();
			}

			return obj->get(p_index, r_valid);

		} break;
		case STRING_NAME: {
		} break;
		case DICTIONARY: {
			const Dictionary *dic = reinterpret_cast<const Dictionary *>(_data._mem);
			const Variant *res = dic->getptr(p_index);
			if (res) {
				valid = true;
				return *res;
			}
		} break;
			DEFAULT_OP_ARRAY_CMD(ARRAY, const Array, ;, return (*arr)[index]) // 20
			DEFAULT_OP_DVECTOR_GET(POOL_BYTE_ARRAY, uint8_t)
			DEFAULT_OP_DVECTOR_GET(POOL_INT_ARRAY, int)
			DEFAULT_OP_DVECTOR_GET(POOL_REAL_ARRAY, real_t)
			DEFAULT_OP_DVECTOR_GET(POOL_STRING_ARRAY, String)
			DEFAULT_OP_DVECTOR_GET(POOL_VECTOR2_ARRAY, Vector2)
			DEFAULT_OP_DVECTOR_GET(POOL_VECTOR2I_ARRAY, Vector2i)
			DEFAULT_OP_DVECTOR_GET(POOL_VECTOR3_ARRAY, Vector3)
			DEFAULT_OP_DVECTOR_GET(POOL_VECTOR3I_ARRAY, Vector3i)
			DEFAULT_OP_DVECTOR_GET(POOL_VECTOR4_ARRAY, Vector4)
			DEFAULT_OP_DVECTOR_GET(POOL_VECTOR4I_ARRAY, Vector4i)
			DEFAULT_OP_DVECTOR_GET(POOL_COLOR_ARRAY, Color)
		default:
			return Variant();
	}

	return Variant();
}

bool Variant::in(const Variant &p_index, bool *r_valid) const {
	if (r_valid) {
		*r_valid = true;
	}

	switch (type) {
		case STRING: {
			if (p_index.get_type() == Variant::STRING) {
				//string index
				String idx = p_index;
				const String *str = reinterpret_cast<const String *>(_data._mem);

				return str->find(idx) != -1;
			}

		} break;
		case OBJECT: {
			Object *obj = _OBJ_PTR(*this);
			if (unlikely(!obj)) {
				if (r_valid) {
					*r_valid = false;
				}
#ifdef DEBUG_ENABLED
				if (_get_obj().rc) {
					ERR_PRINT("Attempted 'in' on a deleted object.");
				}
#endif
				return false;
			}

			bool result;
			obj->get(p_index, &result);
			return result;
		} break;
		case DICTIONARY: {
			const Dictionary *dic = reinterpret_cast<const Dictionary *>(_data._mem);
			return dic->has(p_index);

		} break; // 20
		case ARRAY: {
			const Array *arr = reinterpret_cast<const Array *>(_data._mem);
			int l = arr->size();
			if (l) {
				for (int i = 0; i < l; i++) {
					if (evaluate(OP_EQUAL, (*arr)[i], p_index)) {
						return true;
					}
				}
			}

			return false;

		} break;
		case POOL_BYTE_ARRAY: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				int index = p_index;
				const PoolVector<uint8_t> *arr = reinterpret_cast<const PoolVector<uint8_t> *>(_data._mem);
				int l = arr->size();
				if (l) {
					PoolVector<uint8_t>::Read r = arr->read();
					for (int i = 0; i < l; i++) {
						if (r[i] == index) {
							return true;
						}
					}
				}

				return false;
			}

		} break;
		case POOL_INT_ARRAY: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				int index = p_index;
				const PoolVector<int> *arr = reinterpret_cast<const PoolVector<int> *>(_data._mem);
				int l = arr->size();
				if (l) {
					PoolVector<int>::Read r = arr->read();
					for (int i = 0; i < l; i++) {
						if (r[i] == index) {
							return true;
						}
					}
				}

				return false;
			}
		} break;
		case POOL_REAL_ARRAY: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				real_t index = p_index;
				const PoolVector<real_t> *arr = reinterpret_cast<const PoolVector<real_t> *>(_data._mem);
				int l = arr->size();
				if (l) {
					PoolVector<real_t>::Read r = arr->read();
					for (int i = 0; i < l; i++) {
						if (r[i] == index) {
							return true;
						}
					}
				}

				return false;
			}

		} break;
		case POOL_STRING_ARRAY: {
			if (p_index.get_type() == Variant::STRING) {
				String index = p_index;
				const PoolVector<String> *arr = reinterpret_cast<const PoolVector<String> *>(_data._mem);

				int l = arr->size();
				if (l) {
					PoolVector<String>::Read r = arr->read();
					for (int i = 0; i < l; i++) {
						if (r[i] == index) {
							return true;
						}
					}
				}

				return false;
			}

		} break; //25
		case POOL_VECTOR2_ARRAY: {
			if (p_index.get_type() == Variant::VECTOR2) {
				Vector2 index = p_index;
				const PoolVector<Vector2> *arr = reinterpret_cast<const PoolVector<Vector2> *>(_data._mem);

				int l = arr->size();
				if (l) {
					PoolVector<Vector2>::Read r = arr->read();
					for (int i = 0; i < l; i++) {
						if (r[i] == index) {
							return true;
						}
					}
				}

				return false;
			}

		} break;
		case POOL_VECTOR2I_ARRAY: {
			if (p_index.get_type() == Variant::VECTOR2I) {
				Vector2i index = p_index;
				const PoolVector<Vector2i> *arr = reinterpret_cast<const PoolVector<Vector2i> *>(_data._mem);

				int l = arr->size();
				if (l) {
					PoolVector<Vector2i>::Read r = arr->read();
					for (int i = 0; i < l; i++) {
						if (r[i] == index) {
							return true;
						}
					}
				}

				return false;
			}

		} break;
		case POOL_VECTOR3_ARRAY: {
			if (p_index.get_type() == Variant::VECTOR3) {
				Vector3 index = p_index;
				const PoolVector<Vector3> *arr = reinterpret_cast<const PoolVector<Vector3> *>(_data._mem);

				int l = arr->size();
				if (l) {
					PoolVector<Vector3>::Read r = arr->read();
					for (int i = 0; i < l; i++) {
						if (r[i] == index) {
							return true;
						}
					}
				}

				return false;
			}

		} break;
		case POOL_VECTOR3I_ARRAY: {
			if (p_index.get_type() == Variant::VECTOR3I) {
				Vector3i index = p_index;
				const PoolVector<Vector3i> *arr = reinterpret_cast<const PoolVector<Vector3i> *>(_data._mem);

				int l = arr->size();
				if (l) {
					PoolVector<Vector3i>::Read r = arr->read();
					for (int i = 0; i < l; i++) {
						if (r[i] == index) {
							return true;
						}
					}
				}

				return false;
			}

		} break;
		case POOL_VECTOR4_ARRAY: {
			if (p_index.get_type() == Variant::VECTOR4) {
				Vector4 index = p_index;
				const PoolVector<Vector4> *arr = reinterpret_cast<const PoolVector<Vector4> *>(_data._mem);

				int l = arr->size();
				if (l) {
					PoolVector<Vector4>::Read r = arr->read();
					for (int i = 0; i < l; i++) {
						if (r[i] == index) {
							return true;
						}
					}
				}

				return false;
			}

		} break;
		case POOL_VECTOR4I_ARRAY: {
			if (p_index.get_type() == Variant::VECTOR4I) {
				Vector4i index = p_index;
				const PoolVector<Vector4i> *arr = reinterpret_cast<const PoolVector<Vector4i> *>(_data._mem);

				int l = arr->size();
				if (l) {
					PoolVector<Vector4i>::Read r = arr->read();
					for (int i = 0; i < l; i++) {
						if (r[i] == index) {
							return true;
						}
					}
				}

				return false;
			}

		} break;
		case POOL_COLOR_ARRAY: {
			if (p_index.get_type() == Variant::COLOR) {
				Color index = p_index;
				const PoolVector<Color> *arr = reinterpret_cast<const PoolVector<Color> *>(_data._mem);

				int l = arr->size();
				if (l) {
					PoolVector<Color>::Read r = arr->read();
					for (int i = 0; i < l; i++) {
						if (r[i] == index) {
							return true;
						}
					}
				}

				return false;
			}
		} break;
		default: {
		}
	}

	if (r_valid) {
		*r_valid = false;
	}
	return false;
}

bool Variant::iter_init(Variant &r_iter, bool &valid) const {
	valid = true;
	switch (type) {
		case INT: {
			r_iter = 0;
			return _data._int > 0;
		} break;
		case REAL: {
			r_iter = 0;
			return _data._real > 0.0;
		} break;
		case VECTOR2: {
			int64_t from = reinterpret_cast<const Vector2 *>(_data._mem)->x;
			int64_t to = reinterpret_cast<const Vector2 *>(_data._mem)->y;

			r_iter = from;

			return from < to;
		} break;
		case VECTOR2I: {
			int64_t from = reinterpret_cast<const Vector2i *>(_data._mem)->x;
			int64_t to = reinterpret_cast<const Vector2i *>(_data._mem)->y;

			r_iter = from;

			return from < to;
		} break;
		case VECTOR3: {
			int64_t from = reinterpret_cast<const Vector3 *>(_data._mem)->x;
			int64_t to = reinterpret_cast<const Vector3 *>(_data._mem)->y;
			int64_t step = reinterpret_cast<const Vector3 *>(_data._mem)->z;

			r_iter = from;

			if (from == to) {
				return false;
			} else if (from < to) {
				return step > 0;
			} else {
				return step < 0;
			}
			//return true;
		} break;
		case VECTOR3I: {
			int64_t from = reinterpret_cast<const Vector3i *>(_data._mem)->x;
			int64_t to = reinterpret_cast<const Vector3i *>(_data._mem)->y;
			int64_t step = reinterpret_cast<const Vector3i *>(_data._mem)->z;

			r_iter = from;

			if (from == to) {
				return false;
			} else if (from < to) {
				return step > 0;
			} else {
				return step < 0;
			}
			//return true;
		} break;
		case STRING: {
			const String *str = reinterpret_cast<const String *>(_data._mem);
			if (str->empty()) {
				return false;
			}
			r_iter = 0;
			return true;
		} break;
		case DICTIONARY: {
			const Dictionary *dic = reinterpret_cast<const Dictionary *>(_data._mem);
			if (dic->empty()) {
				return false;
			}

			const Variant *next = dic->next(nullptr);
			r_iter = *next;
			return true;

		} break;
		case ARRAY: {
			const Array *arr = reinterpret_cast<const Array *>(_data._mem);
			if (arr->empty()) {
				return false;
			}
			r_iter = 0;
			return true;
		} break;
		case POOL_BYTE_ARRAY: {
			const PoolVector<uint8_t> *arr = reinterpret_cast<const PoolVector<uint8_t> *>(_data._mem);
			if (arr->size() == 0) {
				return false;
			}
			r_iter = 0;
			return true;

		} break;
		case POOL_INT_ARRAY: {
			const PoolVector<int> *arr = reinterpret_cast<const PoolVector<int> *>(_data._mem);
			if (arr->size() == 0) {
				return false;
			}
			r_iter = 0;
			return true;

		} break;
		case POOL_REAL_ARRAY: {
			const PoolVector<real_t> *arr = reinterpret_cast<const PoolVector<real_t> *>(_data._mem);
			if (arr->size() == 0) {
				return false;
			}
			r_iter = 0;
			return true;

		} break;
		case POOL_STRING_ARRAY: {
			const PoolVector<String> *arr = reinterpret_cast<const PoolVector<String> *>(_data._mem);
			if (arr->size() == 0) {
				return false;
			}
			r_iter = 0;
			return true;
		} break;
		case POOL_VECTOR2_ARRAY: {
			const PoolVector<Vector2> *arr = reinterpret_cast<const PoolVector<Vector2> *>(_data._mem);
			if (arr->size() == 0) {
				return false;
			}
			r_iter = 0;
			return true;
		} break;
		case POOL_VECTOR2I_ARRAY: {
			const PoolVector<Vector2i> *arr = reinterpret_cast<const PoolVector<Vector2i> *>(_data._mem);
			if (arr->size() == 0) {
				return false;
			}
			r_iter = 0;
			return true;
		} break;
		case POOL_VECTOR3_ARRAY: {
			const PoolVector<Vector3> *arr = reinterpret_cast<const PoolVector<Vector3> *>(_data._mem);
			if (arr->size() == 0) {
				return false;
			}
			r_iter = 0;
			return true;
		} break;
		case POOL_VECTOR3I_ARRAY: {
			const PoolVector<Vector3i> *arr = reinterpret_cast<const PoolVector<Vector3i> *>(_data._mem);
			if (arr->size() == 0) {
				return false;
			}
			r_iter = 0;
			return true;
		} break;
		case POOL_VECTOR4_ARRAY: {
			const PoolVector<Vector4> *arr = reinterpret_cast<const PoolVector<Vector4> *>(_data._mem);
			if (arr->size() == 0) {
				return false;
			}
			r_iter = 0;
			return true;
		} break;
		case POOL_VECTOR4I_ARRAY: {
			const PoolVector<Vector4i> *arr = reinterpret_cast<const PoolVector<Vector4i> *>(_data._mem);
			if (arr->size() == 0) {
				return false;
			}
			r_iter = 0;
			return true;
		} break;
		case POOL_COLOR_ARRAY: {
			const PoolVector<Color> *arr = reinterpret_cast<const PoolVector<Color> *>(_data._mem);
			if (arr->size() == 0) {
				return false;
			}
			r_iter = 0;
			return true;

		} break;
		default: {
		}
	}

	valid = false;
	return false;
}
bool Variant::iter_next(Variant &r_iter, bool &valid) const {
	valid = true;
	switch (type) {
		case INT: {
			int64_t idx = r_iter;
			idx++;
			if (idx >= _data._int) {
				return false;
			}
			r_iter = idx;
			return true;
		} break;
		case REAL: {
			int64_t idx = r_iter;
			idx++;
			if (idx >= _data._real) {
				return false;
			}
			r_iter = idx;
			return true;
		} break;
		case VECTOR2: {
			int64_t to = reinterpret_cast<const Vector2 *>(_data._mem)->y;

			int64_t idx = r_iter;
			idx++;

			if (idx >= to) {
				return false;
			}

			r_iter = idx;
			return true;
		} break;
		case VECTOR2I: {
			int64_t to = reinterpret_cast<const Vector2i *>(_data._mem)->y;

			int64_t idx = r_iter;
			idx++;

			if (idx >= to) {
				return false;
			}

			r_iter = idx;
			return true;
		} break;
		case VECTOR3: {
			int64_t to = reinterpret_cast<const Vector3 *>(_data._mem)->y;
			int64_t step = reinterpret_cast<const Vector3 *>(_data._mem)->z;

			int64_t idx = r_iter;
			idx += step;

			if (step < 0 && idx <= to) {
				return false;
			}

			if (step > 0 && idx >= to) {
				return false;
			}

			r_iter = idx;
			return true;
		} break;
		case VECTOR3I: {
			int64_t to = reinterpret_cast<const Vector3i *>(_data._mem)->y;
			int64_t step = reinterpret_cast<const Vector3i *>(_data._mem)->z;

			int64_t idx = r_iter;
			idx += step;

			if (step < 0 && idx <= to) {
				return false;
			}

			if (step > 0 && idx >= to) {
				return false;
			}

			r_iter = idx;
			return true;
		} break;
		case STRING: {
			const String *str = reinterpret_cast<const String *>(_data._mem);
			int idx = r_iter;
			idx++;
			if (idx >= str->length()) {
				return false;
			}
			r_iter = idx;
			return true;
		} break;
		case DICTIONARY: {
			const Dictionary *dic = reinterpret_cast<const Dictionary *>(_data._mem);
			const Variant *next = dic->next(&r_iter);
			if (!next) {
				return false;
			}

			r_iter = *next;
			return true;

		} break;
		case ARRAY: {
			const Array *arr = reinterpret_cast<const Array *>(_data._mem);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size()) {
				return false;
			}
			r_iter = idx;
			return true;
		} break;
		case POOL_BYTE_ARRAY: {
			const PoolVector<uint8_t> *arr = reinterpret_cast<const PoolVector<uint8_t> *>(_data._mem);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size()) {
				return false;
			}
			r_iter = idx;
			return true;

		} break;
		case POOL_INT_ARRAY: {
			const PoolVector<int> *arr = reinterpret_cast<const PoolVector<int> *>(_data._mem);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size()) {
				return false;
			}
			r_iter = idx;
			return true;

		} break;
		case POOL_REAL_ARRAY: {
			const PoolVector<real_t> *arr = reinterpret_cast<const PoolVector<real_t> *>(_data._mem);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size()) {
				return false;
			}
			r_iter = idx;
			return true;

		} break;
		case POOL_STRING_ARRAY: {
			const PoolVector<String> *arr = reinterpret_cast<const PoolVector<String> *>(_data._mem);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size()) {
				return false;
			}
			r_iter = idx;
			return true;
		} break;
		case POOL_VECTOR2_ARRAY: {
			const PoolVector<Vector2> *arr = reinterpret_cast<const PoolVector<Vector2> *>(_data._mem);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size()) {
				return false;
			}
			r_iter = idx;
			return true;
		} break;
		case POOL_VECTOR2I_ARRAY: {
			const PoolVector<Vector2i> *arr = reinterpret_cast<const PoolVector<Vector2i> *>(_data._mem);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size()) {
				return false;
			}
			r_iter = idx;
			return true;
		} break;
		case POOL_VECTOR3_ARRAY: {
			const PoolVector<Vector3> *arr = reinterpret_cast<const PoolVector<Vector3> *>(_data._mem);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size()) {
				return false;
			}
			r_iter = idx;
			return true;
		} break;
		case POOL_VECTOR3I_ARRAY: {
			const PoolVector<Vector3i> *arr = reinterpret_cast<const PoolVector<Vector3i> *>(_data._mem);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size()) {
				return false;
			}
			r_iter = idx;
			return true;
		} break;
		case POOL_VECTOR4_ARRAY: {
			const PoolVector<Vector4> *arr = reinterpret_cast<const PoolVector<Vector4> *>(_data._mem);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size()) {
				return false;
			}
			r_iter = idx;
			return true;
		} break;
		case POOL_VECTOR4I_ARRAY: {
			const PoolVector<Vector4i> *arr = reinterpret_cast<const PoolVector<Vector4i> *>(_data._mem);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size()) {
				return false;
			}
			r_iter = idx;
			return true;
		} break;
		case POOL_COLOR_ARRAY: {
			const PoolVector<Color> *arr = reinterpret_cast<const PoolVector<Color> *>(_data._mem);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size()) {
				return false;
			}
			r_iter = idx;
			return true;
		} break;
		default: {
		}
	}

	valid = false;
	return false;
}

Variant Variant::iter_get(const Variant &r_iter, bool &r_valid) const {
	r_valid = true;
	switch (type) {
		case INT: {
			return r_iter;
		} break;
		case REAL: {
			return r_iter;
		} break;
		case VECTOR2: {
			return r_iter;
		} break;
		case VECTOR2I: {
			return r_iter;
		} break;
		case VECTOR3: {
			return r_iter;
		} break;
		case VECTOR3I: {
			return r_iter;
		} break;
		case STRING: {
			const String *str = reinterpret_cast<const String *>(_data._mem);
			return str->substr(r_iter, 1);
		} break;
		case DICTIONARY: {
			return r_iter; //iterator is the same as the key

		} break;
		case ARRAY: {
			const Array *arr = reinterpret_cast<const Array *>(_data._mem);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case POOL_BYTE_ARRAY: {
			const PoolVector<uint8_t> *arr = reinterpret_cast<const PoolVector<uint8_t> *>(_data._mem);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case POOL_INT_ARRAY: {
			const PoolVector<int> *arr = reinterpret_cast<const PoolVector<int> *>(_data._mem);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case POOL_REAL_ARRAY: {
			const PoolVector<real_t> *arr = reinterpret_cast<const PoolVector<real_t> *>(_data._mem);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case POOL_STRING_ARRAY: {
			const PoolVector<String> *arr = reinterpret_cast<const PoolVector<String> *>(_data._mem);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case POOL_VECTOR2_ARRAY: {
			const PoolVector<Vector2> *arr = reinterpret_cast<const PoolVector<Vector2> *>(_data._mem);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case POOL_VECTOR2I_ARRAY: {
			const PoolVector<Vector2i> *arr = reinterpret_cast<const PoolVector<Vector2i> *>(_data._mem);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case POOL_VECTOR3_ARRAY: {
			const PoolVector<Vector3> *arr = reinterpret_cast<const PoolVector<Vector3> *>(_data._mem);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case POOL_VECTOR3I_ARRAY: {
			const PoolVector<Vector3i> *arr = reinterpret_cast<const PoolVector<Vector3i> *>(_data._mem);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case POOL_VECTOR4_ARRAY: {
			const PoolVector<Vector4> *arr = reinterpret_cast<const PoolVector<Vector4> *>(_data._mem);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case POOL_VECTOR4I_ARRAY: {
			const PoolVector<Vector4i> *arr = reinterpret_cast<const PoolVector<Vector4i> *>(_data._mem);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case POOL_COLOR_ARRAY: {
			const PoolVector<Color> *arr = reinterpret_cast<const PoolVector<Color> *>(_data._mem);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		default: {
		}
	}

	r_valid = false;
	return Variant();
}

Variant Variant::duplicate(bool deep) const {
	switch (type) {
		case OBJECT: {
			/*  breaks stuff :(
			if (deep && !_get_obj().ref.is_null()) {
				Ref<Resource> resource = _get_obj().ref;
				if (resource.is_valid()) {
					return resource->duplicate(true);
				}
			}
			*/
			return *this;
		} break;
		case DICTIONARY:
			return operator Dictionary().duplicate(deep);
		case ARRAY:
			return operator Array().duplicate(deep);
		default:
			return *this;
	}
}

void Variant::sub(const Variant &a, const Variant &b, Variant &r_dst) {
	if (a.type != b.type) {
		return;
	}

	switch (a.type) {
		case NIL: {
			r_dst = Variant();
		}
			return;
		case INT: {
			int64_t va = a._data._int;
			int64_t vb = b._data._int;
			r_dst = int(va - vb);
		}
			return;
		case REAL: {
			real_t ra = a._data._real;
			real_t rb = b._data._real;
			r_dst = ra - rb;
		}
			return;
		case RECT2: {
			const Rect2 *ra = reinterpret_cast<const Rect2 *>(a._data._mem);
			const Rect2 *rb = reinterpret_cast<const Rect2 *>(b._data._mem);
			r_dst = Rect2(ra->position - rb->position, ra->size - rb->size);
		}
			return;
		case RECT2I: {
			const Rect2i *ra = reinterpret_cast<const Rect2i *>(a._data._mem);
			const Rect2i *rb = reinterpret_cast<const Rect2i *>(b._data._mem);

			int32_t vax = ra->position.x;
			int32_t vay = ra->position.y;
			int32_t vbx = ra->size.x;
			int32_t vby = ra->size.y;
			int32_t vcx = rb->position.x;
			int32_t vcy = rb->position.y;
			int32_t vdx = rb->size.x;
			int32_t vdy = rb->size.y;

			r_dst = Rect2i(int32_t(vax - vbx), int32_t(vay - vby), int32_t(vcx - vdx), int32_t(vcy - vdy));
		}
			return;
		case VECTOR2: {
			r_dst = *reinterpret_cast<const Vector2 *>(a._data._mem) - *reinterpret_cast<const Vector2 *>(b._data._mem);
		}
			return;
		case VECTOR2I: {
			int32_t vax = reinterpret_cast<const Vector2i *>(a._data._mem)->x;
			int32_t vbx = reinterpret_cast<const Vector2i *>(b._data._mem)->x;
			int32_t vay = reinterpret_cast<const Vector2i *>(a._data._mem)->y;
			int32_t vby = reinterpret_cast<const Vector2i *>(b._data._mem)->y;
			r_dst = Vector2i(int32_t(vax - vbx), int32_t(vay - vby));
		}
			return;
		case VECTOR3: {
			r_dst = *reinterpret_cast<const Vector3 *>(a._data._mem) - *reinterpret_cast<const Vector3 *>(b._data._mem);
		}
			return;
		case VECTOR3I: {
			int32_t vax = reinterpret_cast<const Vector3i *>(a._data._mem)->x;
			int32_t vbx = reinterpret_cast<const Vector3i *>(b._data._mem)->x;
			int32_t vay = reinterpret_cast<const Vector3i *>(a._data._mem)->y;
			int32_t vby = reinterpret_cast<const Vector3i *>(b._data._mem)->y;
			int32_t vaz = reinterpret_cast<const Vector3i *>(a._data._mem)->z;
			int32_t vbz = reinterpret_cast<const Vector3i *>(b._data._mem)->z;
			r_dst = Vector3i(int32_t(vax - vbx), int32_t(vay - vby), int32_t(vaz - vbz));
		}
			return;
		case VECTOR4: {
			r_dst = *reinterpret_cast<const Vector4 *>(a._data._mem) - *reinterpret_cast<const Vector4 *>(b._data._mem);
		}
			return;
		case VECTOR4I: {
			int32_t vax = reinterpret_cast<const Vector4i *>(a._data._mem)->x;
			int32_t vbx = reinterpret_cast<const Vector4i *>(b._data._mem)->x;
			int32_t vay = reinterpret_cast<const Vector4i *>(a._data._mem)->y;
			int32_t vaw = reinterpret_cast<const Vector4i *>(a._data._mem)->w;
			int32_t vby = reinterpret_cast<const Vector4i *>(b._data._mem)->y;
			int32_t vaz = reinterpret_cast<const Vector4i *>(a._data._mem)->z;
			int32_t vbz = reinterpret_cast<const Vector4i *>(b._data._mem)->z;
			int32_t vbw = reinterpret_cast<const Vector4i *>(b._data._mem)->w;
			r_dst = Vector4i(int32_t(vax - vbx), int32_t(vay - vby), int32_t(vaz - vbz), int32_t(vaw - vbw));
		}
			return;
		case AABB: {
			const ::AABB *ra = reinterpret_cast<const ::AABB *>(a._data._mem);
			const ::AABB *rb = reinterpret_cast<const ::AABB *>(b._data._mem);
			r_dst = ::AABB(ra->position - rb->position, ra->size - rb->size);
		}
			return;
		case QUATERNION: {
			Quaternion empty_rot;
			const Quaternion *qa = reinterpret_cast<const Quaternion *>(a._data._mem);
			const Quaternion *qb = reinterpret_cast<const Quaternion *>(b._data._mem);
			r_dst = (*qb).inverse() * *qa;
		}
			return;
		case COLOR: {
			const Color *ca = reinterpret_cast<const Color *>(a._data._mem);
			const Color *cb = reinterpret_cast<const Color *>(b._data._mem);
			float new_r = ca->r - cb->r;
			float new_g = ca->g - cb->g;
			float new_b = ca->b - cb->b;
			float new_a = ca->a - cb->a;
			new_r = new_r > 1.0 ? 1.0 : new_r;
			new_g = new_g > 1.0 ? 1.0 : new_g;
			new_b = new_b > 1.0 ? 1.0 : new_b;
			new_a = new_a > 1.0 ? 1.0 : new_a;
			r_dst = Color(new_r, new_g, new_b, new_a);
		}
			return;
		default: {
			r_dst = a;
		}
			return;
	}
}

void Variant::blend(const Variant &a, const Variant &b, float c, Variant &r_dst) {
	if (a.type != b.type) {
		if (a.is_num() && b.is_num()) {
			real_t va = a;
			real_t vb = b;
			r_dst = va + vb * c;
		} else {
			r_dst = a;
		}
		return;
	}

	switch (a.type) {
		case NIL: {
			r_dst = Variant();
		}
			return;
		case INT: {
			int64_t va = a._data._int;
			int64_t vb = b._data._int;
			r_dst = int(va + vb * c + 0.5);
		}
			return;
		case REAL: {
			double ra = a._data._real;
			double rb = b._data._real;
			r_dst = ra + rb * c;
		}
			return;
		case RECT2: {
			const Rect2 *ra = reinterpret_cast<const Rect2 *>(a._data._mem);
			const Rect2 *rb = reinterpret_cast<const Rect2 *>(b._data._mem);
			r_dst = Rect2(ra->position + rb->position * c, ra->size + rb->size * c);
		}
			return;
		case RECT2I: {
			const Rect2i *ra = reinterpret_cast<const Rect2i *>(a._data._mem);
			const Rect2i *rb = reinterpret_cast<const Rect2i *>(b._data._mem);
			r_dst = Rect2i(ra->position + rb->position * c, ra->size + rb->size * c);
		}
			return;
		case VECTOR2: {
			r_dst = *reinterpret_cast<const Vector2 *>(a._data._mem) + *reinterpret_cast<const Vector2 *>(b._data._mem) * c;
		}
			return;
		case VECTOR2I: {
			r_dst = *reinterpret_cast<const Vector2i *>(a._data._mem) + *reinterpret_cast<const Vector2i *>(b._data._mem) * c;
		}
			return;
		case VECTOR3: {
			r_dst = *reinterpret_cast<const Vector3 *>(a._data._mem) + *reinterpret_cast<const Vector3 *>(b._data._mem) * c;
		}
			return;
		case VECTOR3I: {
			r_dst = *reinterpret_cast<const Vector3i *>(a._data._mem) + *reinterpret_cast<const Vector3i *>(b._data._mem) * c;
		}
			return;
		case VECTOR4: {
			r_dst = *reinterpret_cast<const Vector4 *>(a._data._mem) + *reinterpret_cast<const Vector4 *>(b._data._mem) * c;
		}
			return;
		case VECTOR4I: {
			r_dst = *reinterpret_cast<const Vector4i *>(a._data._mem) + *reinterpret_cast<const Vector4i *>(b._data._mem) * c;
		}
			return;
		case AABB: {
			const ::AABB *ra = reinterpret_cast<const ::AABB *>(a._data._mem);
			const ::AABB *rb = reinterpret_cast<const ::AABB *>(b._data._mem);
			r_dst = ::AABB(ra->position + rb->position * c, ra->size + rb->size * c);
		}
			return;
		case QUATERNION: {
			Quaternion empty_rot;
			const Quaternion *qa = reinterpret_cast<const Quaternion *>(a._data._mem);
			const Quaternion *qb = reinterpret_cast<const Quaternion *>(b._data._mem);
			r_dst = *qa * empty_rot.slerp(*qb, c);
		}
			return;
		case COLOR: {
			const Color *ca = reinterpret_cast<const Color *>(a._data._mem);
			const Color *cb = reinterpret_cast<const Color *>(b._data._mem);
			float new_r = ca->r + cb->r * c;
			float new_g = ca->g + cb->g * c;
			float new_b = ca->b + cb->b * c;
			float new_a = ca->a + cb->a * c;
			new_r = new_r > 1.0 ? 1.0 : new_r;
			new_g = new_g > 1.0 ? 1.0 : new_g;
			new_b = new_b > 1.0 ? 1.0 : new_b;
			new_a = new_a > 1.0 ? 1.0 : new_a;
			r_dst = Color(new_r, new_g, new_b, new_a);
		}
			return;
		default: {
			r_dst = c < 0.5 ? a : b;
		}
			return;
	}
}

void Variant::interpolate(const Variant &a, const Variant &b, float c, Variant &r_dst) {
	if (a.type != b.type) {
		if (a.is_num() && b.is_num()) {
			//not as efficient but..
			real_t va = a;
			real_t vb = b;
			r_dst = va + (vb - va) * c;

		} else {
			r_dst = a;
		}
		return;
	}

	switch (a.type) {
		case NIL: {
			r_dst = Variant();
		}
			return;
		case BOOL: {
			r_dst = a;
		}
			return;
		case INT: {
			int64_t va = a._data._int;
			int64_t vb = b._data._int;
			r_dst = int(va + (vb - va) * c);
		}
			return;
		case REAL: {
			real_t va = a._data._real;
			real_t vb = b._data._real;
			r_dst = va + (vb - va) * c;
		}
			return;
		case STRING: {
			//this is pretty funny and bizarre, but artists like to use it for typewritter effects
			String sa = *reinterpret_cast<const String *>(a._data._mem);
			String sb = *reinterpret_cast<const String *>(b._data._mem);
			String dst;
			int sa_len = sa.length();
			int sb_len = sb.length();
			int csize = sa_len + (sb_len - sa_len) * c;
			if (csize == 0) {
				r_dst = "";
				return;
			}
			dst.resize(csize + 1);
			dst[csize] = 0;
			int split = csize / 2;

			for (int i = 0; i < csize; i++) {
				CharType chr = ' ';

				if (i < split) {
					if (i < sa.length()) {
						chr = sa[i];
					} else if (i < sb.length()) {
						chr = sb[i];
					}

				} else {
					if (i < sb.length()) {
						chr = sb[i];
					} else if (i < sa.length()) {
						chr = sa[i];
					}
				}

				dst[i] = chr;
			}

			r_dst = dst;
		}
			return;
		case RECT2: {
			r_dst = Rect2(reinterpret_cast<const Rect2 *>(a._data._mem)->position.linear_interpolate(reinterpret_cast<const Rect2 *>(b._data._mem)->position, c), reinterpret_cast<const Rect2 *>(a._data._mem)->size.linear_interpolate(reinterpret_cast<const Rect2 *>(b._data._mem)->size, c));
		}
			return;
		case RECT2I: {
			r_dst = Rect2(reinterpret_cast<const Rect2i *>(a._data._mem)->position.linear_interpolate(reinterpret_cast<const Rect2i *>(b._data._mem)->position, c), reinterpret_cast<const Rect2 *>(a._data._mem)->size.linear_interpolate(reinterpret_cast<const Rect2 *>(b._data._mem)->size, c));
		}
			return;
		case VECTOR2: {
			r_dst = reinterpret_cast<const Vector2 *>(a._data._mem)->linear_interpolate(*reinterpret_cast<const Vector2 *>(b._data._mem), c);
		}
			return;
		case VECTOR2I: {
			r_dst = reinterpret_cast<const Vector2i *>(a._data._mem)->linear_interpolate(*reinterpret_cast<const Vector2i *>(b._data._mem), c);
		}
			return;
		case VECTOR3: {
			r_dst = reinterpret_cast<const Vector3 *>(a._data._mem)->linear_interpolate(*reinterpret_cast<const Vector3 *>(b._data._mem), c);
		}
			return;
		case VECTOR3I: {
			r_dst = reinterpret_cast<const Vector3i *>(a._data._mem)->linear_interpolate(*reinterpret_cast<const Vector3i *>(b._data._mem), c);
		}
			return;
		case VECTOR4: {
			r_dst = reinterpret_cast<const Vector4 *>(a._data._mem)->linear_interpolate(*reinterpret_cast<const Vector4 *>(b._data._mem), c);
		}
			return;
		case VECTOR4I: {
			r_dst = reinterpret_cast<const Vector4i *>(a._data._mem)->linear_interpolate(*reinterpret_cast<const Vector4i *>(b._data._mem), c);
		}
			return;
		case PLANE: {
			r_dst = a;
		}
			return;
		case QUATERNION: {
			r_dst = reinterpret_cast<const Quaternion *>(a._data._mem)->slerp(*reinterpret_cast<const Quaternion *>(b._data._mem), c);
		}
			return;
		case AABB: {
			r_dst = ::AABB(a._data._aabb->position.linear_interpolate(b._data._aabb->position, c), a._data._aabb->size.linear_interpolate(b._data._aabb->size, c));
		}
			return;
		case BASIS: {
			r_dst = Transform(*a._data._basis).interpolate_with(Transform(*b._data._basis), c).basis;
		}
			return;
		case TRANSFORM: {
			r_dst = a._data._transform->interpolate_with(*b._data._transform, c);
		}
			return;
		case TRANSFORM2D: {
			r_dst = a._data._transform2d->interpolate_with(*b._data._transform2d, c);
		}
			return;
		case COLOR: {
			r_dst = reinterpret_cast<const Color *>(a._data._mem)->linear_interpolate(*reinterpret_cast<const Color *>(b._data._mem), c);
		}
			return;
		case OBJECT: {
			r_dst = a;
		}
			return;
		case STRING_NAME: {
			r_dst = a;
		}
			return;
		case DICTIONARY: {
		}
			return;
		case ARRAY: {
			r_dst = a;
		}
			return;
		case POOL_BYTE_ARRAY: {
			r_dst = a;
		}
			return;
		case POOL_INT_ARRAY: {
			const PoolVector<int> *arr_a = reinterpret_cast<const PoolVector<int> *>(a._data._mem);
			const PoolVector<int> *arr_b = reinterpret_cast<const PoolVector<int> *>(b._data._mem);
			int sz = arr_a->size();
			if (sz == 0 || arr_b->size() != sz) {
				r_dst = a;
			} else {
				PoolVector<int> v;
				v.resize(sz);
				{
					PoolVector<int>::Write vw = v.write();
					PoolVector<int>::Read ar = arr_a->read();
					PoolVector<int>::Read br = arr_b->read();

					Variant va;
					for (int i = 0; i < sz; i++) {
						Variant::interpolate(ar[i], br[i], c, va);
						vw[i] = va;
					}
				}
				r_dst = v;
			}
		}
			return;
		case POOL_REAL_ARRAY: {
			const PoolVector<real_t> *arr_a = reinterpret_cast<const PoolVector<real_t> *>(a._data._mem);
			const PoolVector<real_t> *arr_b = reinterpret_cast<const PoolVector<real_t> *>(b._data._mem);
			int sz = arr_a->size();
			if (sz == 0 || arr_b->size() != sz) {
				r_dst = a;
			} else {
				PoolVector<real_t> v;
				v.resize(sz);
				{
					PoolVector<real_t>::Write vw = v.write();
					PoolVector<real_t>::Read ar = arr_a->read();
					PoolVector<real_t>::Read br = arr_b->read();

					Variant va;
					for (int i = 0; i < sz; i++) {
						Variant::interpolate(ar[i], br[i], c, va);
						vw[i] = va;
					}
				}
				r_dst = v;
			}
		}
			return;
		case POOL_STRING_ARRAY: {
			r_dst = a;
		}
			return;
		case POOL_VECTOR2_ARRAY: {
			const PoolVector<Vector2> *arr_a = reinterpret_cast<const PoolVector<Vector2> *>(a._data._mem);
			const PoolVector<Vector2> *arr_b = reinterpret_cast<const PoolVector<Vector2> *>(b._data._mem);
			int sz = arr_a->size();
			if (sz == 0 || arr_b->size() != sz) {
				r_dst = a;
			} else {
				PoolVector<Vector2> v;
				v.resize(sz);
				{
					PoolVector<Vector2>::Write vw = v.write();
					PoolVector<Vector2>::Read ar = arr_a->read();
					PoolVector<Vector2>::Read br = arr_b->read();

					for (int i = 0; i < sz; i++) {
						vw[i] = ar[i].linear_interpolate(br[i], c);
					}
				}
				r_dst = v;
			}
		}
			return;
		case POOL_VECTOR2I_ARRAY: {
			const PoolVector<Vector2i> *arr_a = reinterpret_cast<const PoolVector<Vector2i> *>(a._data._mem);
			const PoolVector<Vector2i> *arr_b = reinterpret_cast<const PoolVector<Vector2i> *>(b._data._mem);
			int sz = arr_a->size();
			if (sz == 0 || arr_b->size() != sz) {
				r_dst = a;
			} else {
				PoolVector<Vector2i> v;
				v.resize(sz);
				{
					PoolVector<Vector2i>::Write vw = v.write();
					PoolVector<Vector2i>::Read ar = arr_a->read();
					PoolVector<Vector2i>::Read br = arr_b->read();

					for (int i = 0; i < sz; i++) {
						vw[i] = ar[i].linear_interpolate(br[i], c);
					}
				}
				r_dst = v;
			}
		}
			return;
		case POOL_VECTOR3_ARRAY: {
			const PoolVector<Vector3> *arr_a = reinterpret_cast<const PoolVector<Vector3> *>(a._data._mem);
			const PoolVector<Vector3> *arr_b = reinterpret_cast<const PoolVector<Vector3> *>(b._data._mem);
			int sz = arr_a->size();
			if (sz == 0 || arr_b->size() != sz) {
				r_dst = a;
			} else {
				PoolVector<Vector3> v;
				v.resize(sz);
				{
					PoolVector<Vector3>::Write vw = v.write();
					PoolVector<Vector3>::Read ar = arr_a->read();
					PoolVector<Vector3>::Read br = arr_b->read();

					for (int i = 0; i < sz; i++) {
						vw[i] = ar[i].linear_interpolate(br[i], c);
					}
				}
				r_dst = v;
			}
		}
			return;
		case POOL_VECTOR3I_ARRAY: {
			const PoolVector<Vector3i> *arr_a = reinterpret_cast<const PoolVector<Vector3i> *>(a._data._mem);
			const PoolVector<Vector3i> *arr_b = reinterpret_cast<const PoolVector<Vector3i> *>(b._data._mem);
			int sz = arr_a->size();
			if (sz == 0 || arr_b->size() != sz) {
				r_dst = a;
			} else {
				PoolVector<Vector3i> v;
				v.resize(sz);
				{
					PoolVector<Vector3i>::Write vw = v.write();
					PoolVector<Vector3i>::Read ar = arr_a->read();
					PoolVector<Vector3i>::Read br = arr_b->read();

					for (int i = 0; i < sz; i++) {
						vw[i] = ar[i].linear_interpolate(br[i], c);
					}
				}
				r_dst = v;
			}
		}
			return;
		case POOL_VECTOR4_ARRAY: {
			const PoolVector<Vector4> *arr_a = reinterpret_cast<const PoolVector<Vector4> *>(a._data._mem);
			const PoolVector<Vector4> *arr_b = reinterpret_cast<const PoolVector<Vector4> *>(b._data._mem);
			int sz = arr_a->size();
			if (sz == 0 || arr_b->size() != sz) {
				r_dst = a;
			} else {
				PoolVector<Vector4> v;
				v.resize(sz);
				{
					PoolVector<Vector4>::Write vw = v.write();
					PoolVector<Vector4>::Read ar = arr_a->read();
					PoolVector<Vector4>::Read br = arr_b->read();

					for (int i = 0; i < sz; i++) {
						vw[i] = ar[i].linear_interpolate(br[i], c);
					}
				}
				r_dst = v;
			}
		}
			return;
		case POOL_VECTOR4I_ARRAY: {
			const PoolVector<Vector4i> *arr_a = reinterpret_cast<const PoolVector<Vector4i> *>(a._data._mem);
			const PoolVector<Vector4i> *arr_b = reinterpret_cast<const PoolVector<Vector4i> *>(b._data._mem);
			int sz = arr_a->size();
			if (sz == 0 || arr_b->size() != sz) {
				r_dst = a;
			} else {
				PoolVector<Vector4i> v;
				v.resize(sz);
				{
					PoolVector<Vector4i>::Write vw = v.write();
					PoolVector<Vector4i>::Read ar = arr_a->read();
					PoolVector<Vector4i>::Read br = arr_b->read();

					for (int i = 0; i < sz; i++) {
						vw[i] = ar[i].linear_interpolate(br[i], c);
					}
				}
				r_dst = v;
			}

			r_dst = a;
		}
			return;
		case POOL_COLOR_ARRAY: {
			const PoolVector<Color> *arr_a = reinterpret_cast<const PoolVector<Color> *>(a._data._mem);
			const PoolVector<Color> *arr_b = reinterpret_cast<const PoolVector<Color> *>(b._data._mem);
			int sz = arr_a->size();
			if (sz == 0 || arr_b->size() != sz) {
				r_dst = a;
			} else {
				PoolVector<Color> v;
				v.resize(sz);
				{
					PoolVector<Color>::Write vw = v.write();
					PoolVector<Color>::Read ar = arr_a->read();
					PoolVector<Color>::Read br = arr_b->read();

					for (int i = 0; i < sz; i++) {
						vw[i] = ar[i].linear_interpolate(br[i], c);
					}
				}
				r_dst = v;
			}
		}
			return;
		default: {
			r_dst = a;
		}
	}
}

static const char *_op_names[Variant::OP_MAX] = {
	"==",
	"!=",
	"<",
	"<=",
	">",
	">=",
	"+",
	"-",
	"*",
	"/",
	"- (negation)",
	"+ (positive)",
	"%",
	"+ (concatenation)",
	"<<",
	">>",
	"&",
	"|",
	"^",
	"~",
	"and",
	"or",
	"xor",
	"not",
	"in"

};

String Variant::get_operator_name(Operator p_op) {
	ERR_FAIL_INDEX_V(p_op, OP_MAX, "");
	return _op_names[p_op];
}

#undef CASE_TYPE_ALL
#undef CASE_TYPE_ALL_BUT_INT
#undef TYPE
#undef TYPES
#undef CASES
#undef SWITCH
#undef SWITCH_OP
#undef CASE_TYPE
#undef CASES
#undef SWITCH
#undef SWITCH_OP
#undef CASE_TYPE
#undef _RETURN
#undef _RETURN_FAIL
#undef DEFAULT_OP_NUM
#undef DEFAULT_OP_NUM_NULL
#undef DEFAULT_OP_NUM_DIV
#undef DEFAULT_OP_NUM_DIV
#undef DEFAULT_OP_NUM_NEG
#undef DEFAULT_OP_NUM_POS
#undef DEFAULT_OP_NUM_VEC
#undef DEFAULT_OP_STR_REV
#undef DEFAULT_OP_STR
#undef DEFAULT_OP_STR_NULL
#undef DEFAULT_OP_STR_NULL_NP
#undef DEFAULT_OP_STR_NULL_SN
#undef DEFAULT_OP_LOCALMEM_REV
#undef DEFAULT_OP_LOCALMEM
#undef DEFAULT_OP_LOCALMEM_NULL
#undef DEFAULT_OP_LOCALMEM_NEG
#undef DEFAULT_OP_LOCALMEM_POS
#undef DEFAULT_OP_LOCALMEM_NUM
#undef DEFAULT_OP_PTR
#undef DEFAULT_OP_PTRREF
#undef DEFAULT_OP_PTRREF_NULL
#undef DEFAULT_OP_ARRAY_EQ
#undef DEFAULT_OP_ARRAY_NEQ
#undef DEFAULT_OP_ARRAY_LT
#undef DEFAULT_OP_ARRAY_GT
#undef DEFAULT_OP_ARRAY_OP
#undef DEFAULT_OP_ARRAY_OP_BODY
#undef DEFAULT_OP_ARRAY_ADD
#undef DEFAULT_OP_ARRAY_CMD
#undef DEFAULT_OP_DVECTOR_SET
#undef DEFAULT_OP_DVECTOR_GET
#line 0
