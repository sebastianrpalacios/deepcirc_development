module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _12, _13, _14, _15, _16, _17, _5, _6, _7, _8, _9;
assign _15 = ~_0;
assign _12 = ~_1;
assign _13 = ~_2;
assign _14 = ~_3;
assign _5 = ~(_12 | _13);
assign _6 = ~(_1 | _14);
assign _16 = ~_6;
assign _7 = ~(_6 | _15);
assign _8 = ~(_0 | _16);
assign _9 = ~(_5 | _7);
assign _17 = ~_9;
assign _4 = _8 | _17;
endmodule
