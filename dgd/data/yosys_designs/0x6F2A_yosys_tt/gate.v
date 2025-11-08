module gate(_0, _1, _2, _3, _828);
input _0, _1, _2, _3;
output _828;
wire _820, _821, _822, _823, _824, _825;
assign _821 = ~(_1 | _2);
assign _820 = ~_3;
assign _822 = ~(_3 | _821);
assign _823 = ~(_2 | _820);
assign _824 = ~(_1 | _823);
assign _825 = ~(_0 | _824);
assign _828 = _822 | _825;
endmodule
