module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _10, _11, _12, _13, _14, _15, _5, _6, _7, _8, _9;
assign _12 = ~_0;
assign _11 = ~_1;
assign _13 = ~_2;
assign _15 = ~_2;
assign _5 = ~(_3 | _11);
assign _6 = ~(_12 | _13);
assign _8 = ~(_1 | _15);
assign _14 = ~_6;
assign _9 = ~(_0 | _8);
assign _7 = ~(_5 | _14);
assign _10 = ~(_7 | _9);
assign _4 = _10;
endmodule
