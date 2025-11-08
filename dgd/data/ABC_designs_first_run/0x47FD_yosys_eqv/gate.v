module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _10, _13, _14, _15, _16, _17, _18, _19, _5, _6, _7, _8, _9;
assign _17 = ~_0;
assign _15 = ~_1;
assign _13 = ~_2;
assign _16 = ~_2;
assign _14 = ~_3;
assign _5 = ~(_1 | _13);
assign _7 = ~(_15 | _16);
assign _6 = ~(_5 | _14);
assign _18 = ~_7;
assign _8 = ~(_7 | _17);
assign _9 = ~(_0 | _18);
assign _10 = ~(_6 | _8);
assign _19 = ~_10;
assign _4 = _9 | _19;
endmodule
