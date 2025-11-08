module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _11, _12, _13, _14, _5, _6, _7, _8;
assign _11 = ~_1;
assign _12 = ~_2;
assign _13 = ~_3;
assign _5 = ~(_1 | _3);
assign _7 = ~(_11 | _12);
assign _6 = ~(_0 | _5);
assign _14 = ~_7;
assign _8 = ~(_13 | _14);
assign _4 = _6 | _8;
endmodule
