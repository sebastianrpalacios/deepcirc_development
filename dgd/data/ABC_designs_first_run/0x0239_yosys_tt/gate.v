module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _10, _13, _14, _15, _16, _17, _18, _5, _6, _7, _8, _9;
assign _13 = ~_0;
assign _15 = ~_1;
assign _14 = ~_2;
assign _6 = ~(_0 | _2);
assign _7 = ~(_3 | _15);
assign _5 = ~(_13 | _14);
assign _16 = ~_7;
assign _18 = ~_5;
assign _8 = ~(_6 | _16);
assign _17 = ~_8;
assign _10 = ~(_8 | _18);
assign _9 = ~(_5 | _17);
assign _4 = _9 | _10;
endmodule
