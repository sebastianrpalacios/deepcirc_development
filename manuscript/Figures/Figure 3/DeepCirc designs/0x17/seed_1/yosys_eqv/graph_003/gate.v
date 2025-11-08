module gate(_0, _1, _2, _13);
input _0, _1, _2;
output _13;
wire _14, _20, _21, _22, _23, _24;
assign _20 = ~_2;
assign _21 = ~(_1 | _0);
assign _22 = ~(_20 | _21);
assign _23 = ~(_0 | _22);
assign _24 = ~(_1 | _22);
assign _14 = ~(_23 | _24);
assign _13 = _14;
endmodule
