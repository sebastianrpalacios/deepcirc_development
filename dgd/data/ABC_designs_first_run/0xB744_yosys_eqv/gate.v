module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _12, _13, _14, _15, _5, _6, _7, _8, _9;
assign _12 = ~_1;
assign _13 = ~_3;
assign _5 = ~(_2 | _12);
assign _7 = ~(_2 | _13);
assign _6 = ~(_0 | _5);
assign _14 = ~_7;
assign _15 = ~_6;
assign _8 = ~(_6 | _14);
assign _9 = ~(_7 | _15);
assign _4 = _8 | _9;
endmodule
