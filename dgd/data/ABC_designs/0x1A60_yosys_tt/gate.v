module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _10, _11, _14, _15, _16, _17, _5, _6, _7, _8, _9;
assign _14 = ~_0;
assign _5 = ~(_1 | _2);
assign _15 = ~_3;
assign _6 = ~(_0 | _3);
assign _8 = ~(_14 | _15);
assign _7 = ~(_5 | _6);
assign _9 = ~(_1 | _8);
assign _16 = ~_7;
assign _17 = ~_9;
assign _11 = ~(_7 | _9);
assign _10 = ~(_16 | _17);
assign _4 = _10 | _11;
endmodule
