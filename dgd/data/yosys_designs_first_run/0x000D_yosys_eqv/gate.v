module gate(_0, _1, _2, _3, _11);
input _0, _1, _2, _3;
output _11;
wire _10, _4, _5, _6, _7, _8, _9;
assign _6 = ~_0;
assign _4 = ~_1;
assign _5 = ~_2;
assign _8 = ~(_4 | _6);
assign _7 = ~(_3 | _5);
assign _9 = ~_8;
assign _10 = ~(_7 | _9);
assign _11 = _10;
endmodule
