module top (
    input clk,
    input rst_n,
    output [6:0] hex,
    output finish // LED indicator when done
);

    // --- 1. IMAGE ROM & STREAMING LOGIC ---
    reg [7:0] img_mem [0:783];
    reg [9:0] r_ptr;
    reg [7:0] pixel_stream;
    reg stream_valid;

    initial begin
        // -----------------------------------------------------------
        // PASTE YOUR 'image_hex_data.txt' CONTENT HERE
        // -----------------------------------------------------------
        // Example: img_mem[0] = 8'h3F; ...
img_mem[0] = 8'hf9;
img_mem[1] = 8'hb9;
img_mem[2] = 8'ha8;
img_mem[3] = 8'ha1;
img_mem[4] = 8'ha0;
img_mem[5] = 8'h9f;
img_mem[6] = 8'h9f;
img_mem[7] = 8'ha2;
img_mem[8] = 8'ha7;
img_mem[9] = 8'hb1;
img_mem[10] = 8'hc4;
img_mem[11] = 8'hda;
img_mem[12] = 8'he7;
img_mem[13] = 8'hea;
img_mem[14] = 8'hef;
img_mem[15] = 8'hf4;
img_mem[16] = 8'hf2;
img_mem[17] = 8'he8;
img_mem[18] = 8'hd4;
img_mem[19] = 8'hb4;
img_mem[20] = 8'ha0;
img_mem[21] = 8'h9e;
img_mem[22] = 8'ha5;
img_mem[23] = 8'haf;
img_mem[24] = 8'hb5;
img_mem[25] = 8'hbb;
img_mem[26] = 8'hd7;
img_mem[27] = 8'h21;
img_mem[28] = 8'hcc;
img_mem[29] = 8'hab;
img_mem[30] = 8'ha3;
img_mem[31] = 8'h9d;
img_mem[32] = 8'h9c;
img_mem[33] = 8'ha8;
img_mem[34] = 8'hbd;
img_mem[35] = 8'hd0;
img_mem[36] = 8'he5;
img_mem[37] = 8'hf4;
img_mem[38] = 8'h00;
img_mem[39] = 8'h08;
img_mem[40] = 8'h0b;
img_mem[41] = 8'h0b;
img_mem[42] = 8'h0c;
img_mem[43] = 8'h0b;
img_mem[44] = 8'h09;
img_mem[45] = 8'h07;
img_mem[46] = 8'h00;
img_mem[47] = 8'hf7;
img_mem[48] = 8'hdc;
img_mem[49] = 8'hb4;
img_mem[50] = 8'ha0;
img_mem[51] = 8'h9f;
img_mem[52] = 8'ha3;
img_mem[53] = 8'ha6;
img_mem[54] = 8'hb9;
img_mem[55] = 8'he7;
img_mem[56] = 8'hc3;
img_mem[57] = 8'ha8;
img_mem[58] = 8'h9f;
img_mem[59] = 8'ha1;
img_mem[60] = 8'hbf;
img_mem[61] = 8'hda;
img_mem[62] = 8'hea;
img_mem[63] = 8'hf9;
img_mem[64] = 8'h02;
img_mem[65] = 8'h06;
img_mem[66] = 8'h09;
img_mem[67] = 8'h0f;
img_mem[68] = 8'h11;
img_mem[69] = 8'h0d;
img_mem[70] = 8'h0b;
img_mem[71] = 8'h0c;
img_mem[72] = 8'h0b;
img_mem[73] = 8'h06;
img_mem[74] = 8'h07;
img_mem[75] = 8'h02;
img_mem[76] = 8'h02;
img_mem[77] = 8'hfd;
img_mem[78] = 8'hca;
img_mem[79] = 8'ha8;
img_mem[80] = 8'ha6;
img_mem[81] = 8'ha7;
img_mem[82] = 8'ha9;
img_mem[83] = 8'hb9;
img_mem[84] = 8'had;
img_mem[85] = 8'ha6;
img_mem[86] = 8'ha4;
img_mem[87] = 8'hd0;
img_mem[88] = 8'heb;
img_mem[89] = 8'hee;
img_mem[90] = 8'hef;
img_mem[91] = 8'hf7;
img_mem[92] = 8'hfe;
img_mem[93] = 8'h01;
img_mem[94] = 8'h07;
img_mem[95] = 8'h0c;
img_mem[96] = 8'h0d;
img_mem[97] = 8'h0c;
img_mem[98] = 8'h0c;
img_mem[99] = 8'h09;
img_mem[100] = 8'h0b;
img_mem[101] = 8'h09;
img_mem[102] = 8'h08;
img_mem[103] = 8'h08;
img_mem[104] = 8'h09;
img_mem[105] = 8'h11;
img_mem[106] = 8'h01;
img_mem[107] = 8'hc6;
img_mem[108] = 8'hb6;
img_mem[109] = 8'haf;
img_mem[110] = 8'haf;
img_mem[111] = 8'hb2;
img_mem[112] = 8'ha9;
img_mem[113] = 8'ha4;
img_mem[114] = 8'hee;
img_mem[115] = 8'h02;
img_mem[116] = 8'hf1;
img_mem[117] = 8'heb;
img_mem[118] = 8'hf0;
img_mem[119] = 8'hf9;
img_mem[120] = 8'h03;
img_mem[121] = 8'h0b;
img_mem[122] = 8'h0b;
img_mem[123] = 8'h0e;
img_mem[124] = 8'h0f;
img_mem[125] = 8'h0f;
img_mem[126] = 8'h0e;
img_mem[127] = 8'h0f;
img_mem[128] = 8'h12;
img_mem[129] = 8'h11;
img_mem[130] = 8'h0f;
img_mem[131] = 8'h0c;
img_mem[132] = 8'h0f;
img_mem[133] = 8'h13;
img_mem[134] = 8'h13;
img_mem[135] = 8'hf5;
img_mem[136] = 8'hd4;
img_mem[137] = 8'hc2;
img_mem[138] = 8'hb9;
img_mem[139] = 8'hbc;
img_mem[140] = 8'hae;
img_mem[141] = 8'hd5;
img_mem[142] = 8'h11;
img_mem[143] = 8'hf8;
img_mem[144] = 8'hf1;
img_mem[145] = 8'hf2;
img_mem[146] = 8'hfd;
img_mem[147] = 8'h00;
img_mem[148] = 8'h05;
img_mem[149] = 8'h05;
img_mem[150] = 8'h0b;
img_mem[151] = 8'h11;
img_mem[152] = 8'h10;
img_mem[153] = 8'h0f;
img_mem[154] = 8'h0b;
img_mem[155] = 8'h10;
img_mem[156] = 8'h19;
img_mem[157] = 8'h1d;
img_mem[158] = 8'h18;
img_mem[159] = 8'h12;
img_mem[160] = 8'h0d;
img_mem[161] = 8'h19;
img_mem[162] = 8'h24;
img_mem[163] = 8'h24;
img_mem[164] = 8'hfa;
img_mem[165] = 8'hce;
img_mem[166] = 8'hc1;
img_mem[167] = 8'hca;
img_mem[168] = 8'hc0;
img_mem[169] = 8'hf3;
img_mem[170] = 8'h1f;
img_mem[171] = 8'h12;
img_mem[172] = 8'h00;
img_mem[173] = 8'hff;
img_mem[174] = 8'h05;
img_mem[175] = 8'h00;
img_mem[176] = 8'h02;
img_mem[177] = 8'h05;
img_mem[178] = 8'h07;
img_mem[179] = 8'h09;
img_mem[180] = 8'h09;
img_mem[181] = 8'h04;
img_mem[182] = 8'hff;
img_mem[183] = 8'h00;
img_mem[184] = 8'hf1;
img_mem[185] = 8'hdc;
img_mem[186] = 8'hce;
img_mem[187] = 8'hc9;
img_mem[188] = 8'hd0;
img_mem[189] = 8'hdc;
img_mem[190] = 8'h02;
img_mem[191] = 8'h35;
img_mem[192] = 8'h2f;
img_mem[193] = 8'he5;
img_mem[194] = 8'hc4;
img_mem[195] = 8'hc4;
img_mem[196] = 8'hca;
img_mem[197] = 8'h07;
img_mem[198] = 8'h38;
img_mem[199] = 8'h1d;
img_mem[200] = 8'hf9;
img_mem[201] = 8'hf4;
img_mem[202] = 8'hf1;
img_mem[203] = 8'hed;
img_mem[204] = 8'hf5;
img_mem[205] = 8'hf2;
img_mem[206] = 8'hf8;
img_mem[207] = 8'h00;
img_mem[208] = 8'h00;
img_mem[209] = 8'hff;
img_mem[210] = 8'he4;
img_mem[211] = 8'he3;
img_mem[212] = 8'hd5;
img_mem[213] = 8'hd3;
img_mem[214] = 8'hdd;
img_mem[215] = 8'he7;
img_mem[216] = 8'hec;
img_mem[217] = 8'hf1;
img_mem[218] = 8'hf4;
img_mem[219] = 8'h09;
img_mem[220] = 8'h2e;
img_mem[221] = 8'h29;
img_mem[222] = 8'hec;
img_mem[223] = 8'hc8;
img_mem[224] = 8'hee;
img_mem[225] = 8'h32;
img_mem[226] = 8'h0c;
img_mem[227] = 8'hc5;
img_mem[228] = 8'hba;
img_mem[229] = 8'hb8;
img_mem[230] = 8'hbb;
img_mem[231] = 8'hbe;
img_mem[232] = 8'hd0;
img_mem[233] = 8'hde;
img_mem[234] = 8'he5;
img_mem[235] = 8'hf5;
img_mem[236] = 8'hf9;
img_mem[237] = 8'hf9;
img_mem[238] = 8'hf2;
img_mem[239] = 8'hf0;
img_mem[240] = 8'hf0;
img_mem[241] = 8'hec;
img_mem[242] = 8'he3;
img_mem[243] = 8'he5;
img_mem[244] = 8'hee;
img_mem[245] = 8'hf4;
img_mem[246] = 8'h00;
img_mem[247] = 8'h0d;
img_mem[248] = 8'h21;
img_mem[249] = 8'h2a;
img_mem[250] = 8'h2c;
img_mem[251] = 8'hd8;
img_mem[252] = 8'h23;
img_mem[253] = 8'h17;
img_mem[254] = 8'hd7;
img_mem[255] = 8'hd9;
img_mem[256] = 8'hdd;
img_mem[257] = 8'hdd;
img_mem[258] = 8'hd8;
img_mem[259] = 8'hd8;
img_mem[260] = 8'hde;
img_mem[261] = 8'he2;
img_mem[262] = 8'he5;
img_mem[263] = 8'hf0;
img_mem[264] = 8'hfe;
img_mem[265] = 8'h00;
img_mem[266] = 8'hf7;
img_mem[267] = 8'hf4;
img_mem[268] = 8'hed;
img_mem[269] = 8'hd9;
img_mem[270] = 8'hbc;
img_mem[271] = 8'hb2;
img_mem[272] = 8'hb2;
img_mem[273] = 8'hd1;
img_mem[274] = 8'hd0;
img_mem[275] = 8'hf1;
img_mem[276] = 8'h0e;
img_mem[277] = 8'h1f;
img_mem[278] = 8'h32;
img_mem[279] = 8'hfa;
img_mem[280] = 8'h11;
img_mem[281] = 8'hfd;
img_mem[282] = 8'he7;
img_mem[283] = 8'hd8;
img_mem[284] = 8'hc0;
img_mem[285] = 8'hb3;
img_mem[286] = 8'haf;
img_mem[287] = 8'hb9;
img_mem[288] = 8'hd1;
img_mem[289] = 8'hdf;
img_mem[290] = 8'he3;
img_mem[291] = 8'hf6;
img_mem[292] = 8'h0e;
img_mem[293] = 8'h10;
img_mem[294] = 8'h02;
img_mem[295] = 8'hf2;
img_mem[296] = 8'he8;
img_mem[297] = 8'hd2;
img_mem[298] = 8'heb;
img_mem[299] = 8'hcc;
img_mem[300] = 8'hea;
img_mem[301] = 8'hff;
img_mem[302] = 8'hd8;
img_mem[303] = 8'he1;
img_mem[304] = 8'hfe;
img_mem[305] = 8'h1c;
img_mem[306] = 8'h2b;
img_mem[307] = 8'h17;
img_mem[308] = 8'h01;
img_mem[309] = 8'hee;
img_mem[310] = 8'hd4;
img_mem[311] = 8'hb7;
img_mem[312] = 8'he1;
img_mem[313] = 8'hc8;
img_mem[314] = 8'hba;
img_mem[315] = 8'hdf;
img_mem[316] = 8'hc7;
img_mem[317] = 8'hdf;
img_mem[318] = 8'he1;
img_mem[319] = 8'h00;
img_mem[320] = 8'h1b;
img_mem[321] = 8'h1e;
img_mem[322] = 8'h10;
img_mem[323] = 8'h07;
img_mem[324] = 8'hfc;
img_mem[325] = 8'he9;
img_mem[326] = 8'he5;
img_mem[327] = 8'hea;
img_mem[328] = 8'hf4;
img_mem[329] = 8'hf9;
img_mem[330] = 8'h05;
img_mem[331] = 8'h14;
img_mem[332] = 8'h24;
img_mem[333] = 8'h2d;
img_mem[334] = 8'h32;
img_mem[335] = 8'h35;
img_mem[336] = 8'h05;
img_mem[337] = 8'hea;
img_mem[338] = 8'hd1;
img_mem[339] = 8'hc9;
img_mem[340] = 8'hdc;
img_mem[341] = 8'he4;
img_mem[342] = 8'hde;
img_mem[343] = 8'he1;
img_mem[344] = 8'he6;
img_mem[345] = 8'heb;
img_mem[346] = 8'hef;
img_mem[347] = 8'h07;
img_mem[348] = 8'h1f;
img_mem[349] = 8'h24;
img_mem[350] = 8'h16;
img_mem[351] = 8'h10;
img_mem[352] = 8'h0f;
img_mem[353] = 8'h0e;
img_mem[354] = 8'h05;
img_mem[355] = 8'h00;
img_mem[356] = 8'h07;
img_mem[357] = 8'h10;
img_mem[358] = 8'h1a;
img_mem[359] = 8'h27;
img_mem[360] = 8'h33;
img_mem[361] = 8'h3f;
img_mem[362] = 8'h3e;
img_mem[363] = 8'h4a;
img_mem[364] = 8'h10;
img_mem[365] = 8'h0b;
img_mem[366] = 8'h00;
img_mem[367] = 8'hf7;
img_mem[368] = 8'hf3;
img_mem[369] = 8'hf6;
img_mem[370] = 8'hf6;
img_mem[371] = 8'hf6;
img_mem[372] = 8'hf8;
img_mem[373] = 8'hf4;
img_mem[374] = 8'hf5;
img_mem[375] = 8'h0a;
img_mem[376] = 8'h26;
img_mem[377] = 8'h2a;
img_mem[378] = 8'h23;
img_mem[379] = 8'h17;
img_mem[380] = 8'h0d;
img_mem[381] = 8'h18;
img_mem[382] = 8'h1c;
img_mem[383] = 8'h1b;
img_mem[384] = 8'h19;
img_mem[385] = 8'h1b;
img_mem[386] = 8'h21;
img_mem[387] = 8'h28;
img_mem[388] = 8'h2c;
img_mem[389] = 8'h31;
img_mem[390] = 8'h37;
img_mem[391] = 8'h4c;
img_mem[392] = 8'h26;
img_mem[393] = 8'h2f;
img_mem[394] = 8'h1f;
img_mem[395] = 8'h11;
img_mem[396] = 8'h0b;
img_mem[397] = 8'h08;
img_mem[398] = 8'h07;
img_mem[399] = 8'h07;
img_mem[400] = 8'hff;
img_mem[401] = 8'hf2;
img_mem[402] = 8'h00;
img_mem[403] = 8'h15;
img_mem[404] = 8'h23;
img_mem[405] = 8'h2c;
img_mem[406] = 8'h32;
img_mem[407] = 8'h2f;
img_mem[408] = 8'h1d;
img_mem[409] = 8'h0f;
img_mem[410] = 8'h1c;
img_mem[411] = 8'h24;
img_mem[412] = 8'h26;
img_mem[413] = 8'h25;
img_mem[414] = 8'h25;
img_mem[415] = 8'h24;
img_mem[416] = 8'h23;
img_mem[417] = 8'h24;
img_mem[418] = 8'h29;
img_mem[419] = 8'h48;
img_mem[420] = 8'h27;
img_mem[421] = 8'h1f;
img_mem[422] = 8'h13;
img_mem[423] = 8'h10;
img_mem[424] = 8'h10;
img_mem[425] = 8'h11;
img_mem[426] = 8'h0e;
img_mem[427] = 8'h0a;
img_mem[428] = 8'h02;
img_mem[429] = 8'h18;
img_mem[430] = 8'h08;
img_mem[431] = 8'hff;
img_mem[432] = 8'h09;
img_mem[433] = 8'h0c;
img_mem[434] = 8'h07;
img_mem[435] = 8'h12;
img_mem[436] = 8'h22;
img_mem[437] = 8'h23;
img_mem[438] = 8'h0e;
img_mem[439] = 8'h1c;
img_mem[440] = 8'h27;
img_mem[441] = 8'h29;
img_mem[442] = 8'h28;
img_mem[443] = 8'h25;
img_mem[444] = 8'h1e;
img_mem[445] = 8'h25;
img_mem[446] = 8'h1a;
img_mem[447] = 8'h39;
img_mem[448] = 8'h15;
img_mem[449] = 8'h03;
img_mem[450] = 8'h05;
img_mem[451] = 8'h09;
img_mem[452] = 8'h11;
img_mem[453] = 8'h13;
img_mem[454] = 8'h14;
img_mem[455] = 8'h0d;
img_mem[456] = 8'h00;
img_mem[457] = 8'h00;
img_mem[458] = 8'hd4;
img_mem[459] = 8'hc4;
img_mem[460] = 8'hec;
img_mem[461] = 8'hf4;
img_mem[462] = 8'hd0;
img_mem[463] = 8'hc3;
img_mem[464] = 8'he8;
img_mem[465] = 8'h0e;
img_mem[466] = 8'h1f;
img_mem[467] = 8'h15;
img_mem[468] = 8'h1e;
img_mem[469] = 8'h22;
img_mem[470] = 8'h24;
img_mem[471] = 8'h27;
img_mem[472] = 8'h28;
img_mem[473] = 8'h22;
img_mem[474] = 8'h1c;
img_mem[475] = 8'h30;
img_mem[476] = 8'h0c;
img_mem[477] = 8'hf8;
img_mem[478] = 8'h02;
img_mem[479] = 8'h03;
img_mem[480] = 8'h0a;
img_mem[481] = 8'h12;
img_mem[482] = 8'h12;
img_mem[483] = 8'h0a;
img_mem[484] = 8'hf5;
img_mem[485] = 8'hee;
img_mem[486] = 8'hde;
img_mem[487] = 8'hdf;
img_mem[488] = 8'heb;
img_mem[489] = 8'hf7;
img_mem[490] = 8'h03;
img_mem[491] = 8'hfb;
img_mem[492] = 8'hfa;
img_mem[493] = 8'h21;
img_mem[494] = 8'h3b;
img_mem[495] = 8'h45;
img_mem[496] = 8'h19;
img_mem[497] = 8'h1c;
img_mem[498] = 8'h1e;
img_mem[499] = 8'h20;
img_mem[500] = 8'h1e;
img_mem[501] = 8'h1a;
img_mem[502] = 8'h27;
img_mem[503] = 8'h2a;
img_mem[504] = 8'h14;
img_mem[505] = 8'hfd;
img_mem[506] = 8'h00;
img_mem[507] = 8'h03;
img_mem[508] = 8'h07;
img_mem[509] = 8'h0e;
img_mem[510] = 8'h0e;
img_mem[511] = 8'h07;
img_mem[512] = 8'h0b;
img_mem[513] = 8'h2e;
img_mem[514] = 8'h2e;
img_mem[515] = 8'hfd;
img_mem[516] = 8'h09;
img_mem[517] = 8'h05;
img_mem[518] = 8'h0e;
img_mem[519] = 8'h0a;
img_mem[520] = 8'h19;
img_mem[521] = 8'h20;
img_mem[522] = 8'h2f;
img_mem[523] = 8'h37;
img_mem[524] = 8'h15;
img_mem[525] = 8'h15;
img_mem[526] = 8'h16;
img_mem[527] = 8'h18;
img_mem[528] = 8'h1c;
img_mem[529] = 8'h1e;
img_mem[530] = 8'h28;
img_mem[531] = 8'h37;
img_mem[532] = 8'h0c;
img_mem[533] = 8'h00;
img_mem[534] = 8'h01;
img_mem[535] = 8'hfe;
img_mem[536] = 8'h02;
img_mem[537] = 8'h06;
img_mem[538] = 8'h04;
img_mem[539] = 8'h03;
img_mem[540] = 8'h14;
img_mem[541] = 8'h1b;
img_mem[542] = 8'h29;
img_mem[543] = 8'h07;
img_mem[544] = 8'hf3;
img_mem[545] = 8'hfd;
img_mem[546] = 8'hed;
img_mem[547] = 8'h00;
img_mem[548] = 8'h17;
img_mem[549] = 8'h20;
img_mem[550] = 8'h22;
img_mem[551] = 8'h20;
img_mem[552] = 8'h0f;
img_mem[553] = 8'h0e;
img_mem[554] = 8'h11;
img_mem[555] = 8'h14;
img_mem[556] = 8'h19;
img_mem[557] = 8'h1b;
img_mem[558] = 8'h25;
img_mem[559] = 8'h44;
img_mem[560] = 8'h16;
img_mem[561] = 8'h0a;
img_mem[562] = 8'h02;
img_mem[563] = 8'hf9;
img_mem[564] = 8'hfe;
img_mem[565] = 8'hfc;
img_mem[566] = 8'hff;
img_mem[567] = 8'hfc;
img_mem[568] = 8'h0e;
img_mem[569] = 8'h08;
img_mem[570] = 8'h19;
img_mem[571] = 8'hf5;
img_mem[572] = 8'hd5;
img_mem[573] = 8'hd6;
img_mem[574] = 8'he7;
img_mem[575] = 8'h02;
img_mem[576] = 8'h0f;
img_mem[577] = 8'h17;
img_mem[578] = 8'h1c;
img_mem[579] = 8'h02;
img_mem[580] = 8'hfc;
img_mem[581] = 8'h02;
img_mem[582] = 8'h0e;
img_mem[583] = 8'h21;
img_mem[584] = 8'h14;
img_mem[585] = 8'h18;
img_mem[586] = 8'h2c;
img_mem[587] = 8'h4d;
img_mem[588] = 8'h19;
img_mem[589] = 8'h19;
img_mem[590] = 8'h06;
img_mem[591] = 8'hf4;
img_mem[592] = 8'hf8;
img_mem[593] = 8'hf7;
img_mem[594] = 8'hf7;
img_mem[595] = 8'hf9;
img_mem[596] = 8'h00;
img_mem[597] = 8'h06;
img_mem[598] = 8'h0d;
img_mem[599] = 8'he6;
img_mem[600] = 8'hc7;
img_mem[601] = 8'hd8;
img_mem[602] = 8'h0f;
img_mem[603] = 8'h11;
img_mem[604] = 8'h15;
img_mem[605] = 8'h14;
img_mem[606] = 8'h00;
img_mem[607] = 8'hc9;
img_mem[608] = 8'he7;
img_mem[609] = 8'hfc;
img_mem[610] = 8'h24;
img_mem[611] = 8'h45;
img_mem[612] = 8'h46;
img_mem[613] = 8'h18;
img_mem[614] = 8'h31;
img_mem[615] = 8'h55;
img_mem[616] = 8'h3e;
img_mem[617] = 8'h2a;
img_mem[618] = 8'h0c;
img_mem[619] = 8'hfd;
img_mem[620] = 8'hf7;
img_mem[621] = 8'hf5;
img_mem[622] = 8'hf2;
img_mem[623] = 8'hfc;
img_mem[624] = 8'hfd;
img_mem[625] = 8'h06;
img_mem[626] = 8'h0a;
img_mem[627] = 8'he7;
img_mem[628] = 8'hf2;
img_mem[629] = 8'h0e;
img_mem[630] = 8'h1f;
img_mem[631] = 8'h24;
img_mem[632] = 8'h1e;
img_mem[633] = 8'h18;
img_mem[634] = 8'he9;
img_mem[635] = 8'he5;
img_mem[636] = 8'hea;
img_mem[637] = 8'h08;
img_mem[638] = 8'h29;
img_mem[639] = 8'h3a;
img_mem[640] = 8'h35;
img_mem[641] = 8'h1a;
img_mem[642] = 8'h32;
img_mem[643] = 8'h4b;
img_mem[644] = 8'h4f;
img_mem[645] = 8'h4a;
img_mem[646] = 8'h24;
img_mem[647] = 8'h02;
img_mem[648] = 8'hf8;
img_mem[649] = 8'hf2;
img_mem[650] = 8'hf0;
img_mem[651] = 8'h10;
img_mem[652] = 8'h09;
img_mem[653] = 8'h10;
img_mem[654] = 8'h04;
img_mem[655] = 8'hd0;
img_mem[656] = 8'hf9;
img_mem[657] = 8'h1d;
img_mem[658] = 8'h29;
img_mem[659] = 8'h27;
img_mem[660] = 8'h20;
img_mem[661] = 8'hfd;
img_mem[662] = 8'he0;
img_mem[663] = 8'he8;
img_mem[664] = 8'h04;
img_mem[665] = 8'h1b;
img_mem[666] = 8'h2f;
img_mem[667] = 8'h31;
img_mem[668] = 8'h16;
img_mem[669] = 8'h20;
img_mem[670] = 8'h34;
img_mem[671] = 8'h4c;
img_mem[672] = 8'h4b;
img_mem[673] = 8'h4b;
img_mem[674] = 8'h46;
img_mem[675] = 8'h15;
img_mem[676] = 8'hfe;
img_mem[677] = 8'hf2;
img_mem[678] = 8'hff;
img_mem[679] = 8'h13;
img_mem[680] = 8'h16;
img_mem[681] = 8'h11;
img_mem[682] = 8'hf8;
img_mem[683] = 8'he1;
img_mem[684] = 8'h18;
img_mem[685] = 8'h23;
img_mem[686] = 8'h2c;
img_mem[687] = 8'h27;
img_mem[688] = 8'h19;
img_mem[689] = 8'hd7;
img_mem[690] = 8'he6;
img_mem[691] = 8'h13;
img_mem[692] = 8'h25;
img_mem[693] = 8'h27;
img_mem[694] = 8'h27;
img_mem[695] = 8'h1a;
img_mem[696] = 8'h10;
img_mem[697] = 8'h26;
img_mem[698] = 8'h40;
img_mem[699] = 8'h4d;
img_mem[700] = 8'h50;
img_mem[701] = 8'h50;
img_mem[702] = 8'h50;
img_mem[703] = 8'h34;
img_mem[704] = 8'h0b;
img_mem[705] = 8'hf7;
img_mem[706] = 8'h1a;
img_mem[707] = 8'h16;
img_mem[708] = 8'h1d;
img_mem[709] = 8'h15;
img_mem[710] = 8'hf1;
img_mem[711] = 8'h08;
img_mem[712] = 8'h14;
img_mem[713] = 8'h1d;
img_mem[714] = 8'h2b;
img_mem[715] = 8'h25;
img_mem[716] = 8'hf1;
img_mem[717] = 8'heb;
img_mem[718] = 8'h1e;
img_mem[719] = 8'h32;
img_mem[720] = 8'h32;
img_mem[721] = 8'h2a;
img_mem[722] = 8'h19;
img_mem[723] = 8'h05;
img_mem[724] = 8'h16;
img_mem[725] = 8'h32;
img_mem[726] = 8'h48;
img_mem[727] = 8'h4d;
img_mem[728] = 8'h50;
img_mem[729] = 8'h50;
img_mem[730] = 8'h4f;
img_mem[731] = 8'h4e;
img_mem[732] = 8'h2a;
img_mem[733] = 8'h0f;
img_mem[734] = 8'h0c;
img_mem[735] = 8'h0e;
img_mem[736] = 8'h17;
img_mem[737] = 8'h02;
img_mem[738] = 8'hf4;
img_mem[739] = 8'hfc;
img_mem[740] = 8'h07;
img_mem[741] = 8'h0c;
img_mem[742] = 8'h1b;
img_mem[743] = 8'h04;
img_mem[744] = 8'hf1;
img_mem[745] = 8'h1b;
img_mem[746] = 8'h2f;
img_mem[747] = 8'h33;
img_mem[748] = 8'h2d;
img_mem[749] = 8'h26;
img_mem[750] = 8'h02;
img_mem[751] = 8'h06;
img_mem[752] = 8'h11;
img_mem[753] = 8'h3b;
img_mem[754] = 8'h43;
img_mem[755] = 8'h49;
img_mem[756] = 8'h4c;
img_mem[757] = 8'h4d;
img_mem[758] = 8'h4d;
img_mem[759] = 8'h4c;
img_mem[760] = 8'h4c;
img_mem[761] = 8'h1d;
img_mem[762] = 8'h00;
img_mem[763] = 8'h02;
img_mem[764] = 8'h06;
img_mem[765] = 8'heb;
img_mem[766] = 8'hf6;
img_mem[767] = 8'hfd;
img_mem[768] = 8'h01;
img_mem[769] = 8'h08;
img_mem[770] = 8'h0a;
img_mem[771] = 8'hf6;
img_mem[772] = 8'h0e;
img_mem[773] = 8'h1f;
img_mem[774] = 8'h30;
img_mem[775] = 8'h2f;
img_mem[776] = 8'h29;
img_mem[777] = 8'hfc;
img_mem[778] = 8'hf2;
img_mem[779] = 8'hf8;
img_mem[780] = 8'h04;
img_mem[781] = 8'h30;
img_mem[782] = 8'h43;
img_mem[783] = 8'h4c;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_ptr <= 0;
            stream_valid <= 0;
            pixel_stream <= 0;
        end else begin
            // Stream the 784 pixels one by one
            if (r_ptr < 784) begin
                pixel_stream <= img_mem[r_ptr];
                stream_valid <= 1;
                r_ptr <= r_ptr + 1;
            end else begin
                stream_valid <= 0;
            end
        end
    end

    // --- 2. LAYER 1 (28x28 -> 13x13) ---
    wire signed [11:0] l1_out0, l1_out1, l1_out2;
    wire l1_valid;

    layer1 L1 (
        .clk(clk),
        .rst_n(rst_n),
        .pixel_in(pixel_stream),
        .valid_in(stream_valid),
        .out0(l1_out0),
        .out1(l1_out1),
        .out2(l1_out2),
        .valid_out(l1_valid)
    );

    // --- 3. LAYER 2 (13x13 -> 5x5) ---
    wire signed [11:0] l2_out0, l2_out1, l2_out2;
    wire l2_valid;

    layer2 L2 (
        .clk(clk),
        .rst_n(rst_n),
        .in0(l1_out0),
        .in1(l1_out1),
        .in2(l1_out2),
        .valid_in(l1_valid),
        .out0(l2_out0),
        .out1(l2_out1),
        .out2(l2_out2),
        .valid_out(l2_valid)
    );

    // --- 4. LAYER 3 (5x5 -> 1x1) ---
    wire signed [11:0] l3_out0, l3_out1, l3_out2;
    wire l3_valid;

    layer3 L3 (
        .clk(clk),
        .rst_n(rst_n),
        .in0(l2_out0),
        .in1(l2_out1),
        .in2(l2_out2),
        .valid_in(l2_valid),
        .out0(l3_out0),
        .out1(l3_out1),
        .out2(l3_out2),
        .valid_out(l3_valid)
    );

    // --- 5. FULLY CONNECTED (Classifier) ---
    wire signed [11:0] fc_score;
    wire fc_valid;

    fully_connected FC (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(l3_valid),
        .in0(l3_out0),
        .in1(l3_out1),
        .in2(l3_out2),
        .data_out(fc_score),
        .valid_out(fc_valid)
    );

    // --- 6. COMPARATOR (Decider) ---
    wire [2:0] decision;
    wire comp_done;

    comparator COMP (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(fc_valid),
        .data_in(fc_score),
        .decision(decision),
        .valid_out(comp_done)
    );

    // --- 7. HEX DISPLAY (Output) ---
    hexdigit HEX (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(comp_done),
        .decision(decision),
        .hex(hex),
        .valid_out(finish)
    );

endmodule