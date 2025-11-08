module gate(_0, _1, _2, _13);
input _0, _1, _2;
output _13;
wire _20, _21, _22, _23, _24;
assign _20 = ~_1;
assign _21 = ~(_0 | _20);
assign _22 = ~(_20 | _21);
assign _23 = ~(_0 | _21);
assign _24 = ~(_2 | _23);
assign _13 = _22 | _24;
endmodule
