module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _10, _11, _14, _15, _16, _17, _18, _5, _6, _7, _8, _9;
assign _17 = ~_0;
assign _15 = ~_2;
assign _14 = ~_3;
assign _6 = ~(_1 | _15);
assign _5 = ~(_2 | _14);
assign _16 = ~_6;
assign _10 = ~(_5 | _17);
assign _7 = ~(_3 | _16);
assign _18 = ~_10;
assign _8 = ~(_5 | _7);
assign _11 = ~(_6 | _18);
assign _9 = ~(_0 | _8);
assign _4 = _9 | _11;
endmodule
