module gate(_0, _1, _2, _13);
input _0, _1, _2;
output _13;
wire _20, _21, _22, _23, _24, _25;
assign _20 = ~_0;
assign _24 = ~_2;
assign _21 = ~(_1 | _20);
assign _22 = ~(_20 | _21);
assign _23 = ~(_1 | _21);
assign _25 = ~(_23 | _24);
assign _13 = _22 | _25;
endmodule
