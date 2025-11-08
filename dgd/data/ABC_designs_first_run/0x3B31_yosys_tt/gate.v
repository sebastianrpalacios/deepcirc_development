module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _10, _11, _12, _5, _6, _7;
assign _10 = ~_1;
assign _11 = ~_2;
assign _5 = ~(_3 | _10);
assign _12 = ~_5;
assign _6 = ~(_5 | _11);
assign _7 = ~(_0 | _12);
assign _4 = _6 | _7;
endmodule
