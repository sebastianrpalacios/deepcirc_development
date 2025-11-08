module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _10, _5, _6, _7, _8, _9;
assign _8 = ~_0;
assign _10 = ~_1;
assign _5 = ~(_2 | _8);
assign _9 = ~_5;
assign _6 = ~(_3 | _9);
assign _7 = ~(_6 | _10);
assign _4 = _7;
endmodule
