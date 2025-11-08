module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _10, _13, _14, _15, _16, _17, _18, _19, _5, _6, _7, _8, _9;
assign _17 = ~_0;
assign _13 = ~_1;
assign _18 = ~_1;
assign _14 = ~_2;
assign _15 = ~_3;
assign _8 = ~(_3 | _17);
assign _5 = ~(_0 | _13);
assign _9 = ~(_2 | _18);
assign _6 = ~(_14 | _15);
assign _10 = ~(_8 | _9);
assign _16 = ~_6;
assign _19 = ~_10;
assign _7 = ~(_5 | _16);
assign _4 = _7 | _19;
endmodule
