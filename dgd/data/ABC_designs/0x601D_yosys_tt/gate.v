module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _10, _11, _12, _15, _16, _17, _18, _19, _5, _6, _7, _8, _9;
assign _15 = ~_0;
assign _16 = ~_2;
assign _7 = ~(_1 | _2);
assign _5 = ~(_1 | _3);
assign _8 = ~(_15 | _16);
assign _6 = ~(_0 | _5);
assign _17 = ~_8;
assign _18 = ~_6;
assign _9 = ~(_3 | _17);
assign _10 = ~(_7 | _9);
assign _19 = ~_10;
assign _11 = ~(_10 | _18);
assign _12 = ~(_6 | _19);
assign _4 = _11 | _12;
endmodule
