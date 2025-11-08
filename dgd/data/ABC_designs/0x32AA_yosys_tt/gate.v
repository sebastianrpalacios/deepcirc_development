module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _5, _6, _7, _8, _9;
assign _6 = ~(_0 | _1);
assign _5 = ~(_0 | _2);
assign _9 = ~_3;
assign _7 = ~(_6 | _9);
assign _8 = ~(_5 | _7);
assign _4 = _8;
endmodule
