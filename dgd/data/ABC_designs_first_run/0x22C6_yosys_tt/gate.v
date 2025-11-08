module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _10, _11, _12, _13, _14, _15, _5, _6, _7, _8, _9;
assign _12 = ~_0;
assign _15 = ~_0;
assign _14 = ~_1;
assign _13 = ~_2;
assign _5 = ~(_1 | _12);
assign _8 = ~(_3 | _14);
assign _6 = ~(_3 | _5);
assign _9 = ~(_8 | _15);
assign _7 = ~(_6 | _13);
assign _10 = ~(_2 | _9);
assign _11 = ~(_7 | _10);
assign _4 = _11;
endmodule
