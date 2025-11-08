module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _10, _11, _12, _5, _6, _7, _8, _9;
assign _5 = ~(_1 | _2);
assign _6 = ~(_0 | _2);
assign _11 = ~_5;
assign _7 = ~(_3 | _6);
assign _12 = ~_7;
assign _8 = ~(_7 | _11);
assign _9 = ~(_5 | _12);
assign _10 = ~(_8 | _9);
assign _4 = _10;
endmodule
