module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _10, _13, _14, _15, _16, _17, _18, _5, _6, _7, _8, _9;
assign _16 = ~_1;
assign _13 = ~_2;
assign _14 = ~_3;
assign _15 = ~_3;
assign _8 = ~(_0 | _16);
assign _5 = ~(_13 | _14);
assign _6 = ~(_0 | _15);
assign _17 = ~_8;
assign _7 = ~(_1 | _6);
assign _9 = ~(_2 | _17);
assign _10 = ~(_5 | _7);
assign _18 = ~_10;
assign _4 = _9 | _18;
endmodule
