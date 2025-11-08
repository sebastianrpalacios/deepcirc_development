module gate(_0, _1, _2, _3, _52);
input _0, _1, _2, _3;
output _52;
wire _47, _48, _49, _50, _51;
assign _50 = ~(_0 | _1);
assign _47 = ~_2;
assign _48 = ~(_1 | _47);
assign _49 = ~(_3 | _48);
assign _51 = ~(_49 | _50);
assign _52 = _51;
endmodule
