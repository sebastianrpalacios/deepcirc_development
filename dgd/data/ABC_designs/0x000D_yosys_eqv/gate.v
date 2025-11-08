module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _10, _11, _5, _6, _7, _8, _9;
assign _9 = ~_0;
assign _10 = ~_1;
assign _8 = ~_2;
assign _6 = ~(_9 | _10);
assign _5 = ~(_3 | _8);
assign _11 = ~_6;
assign _7 = ~(_5 | _11);
assign _4 = _7;
endmodule
