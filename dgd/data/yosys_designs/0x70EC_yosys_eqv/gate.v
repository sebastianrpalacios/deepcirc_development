module gate(_0, _1, _2, _3, _856);
input _0, _1, _2, _3;
output _856;
wire _846, _847, _848, _849, _850, _851, _852, _853;
assign _848 = ~_0;
assign _847 = ~_2;
assign _846 = ~_3;
assign _849 = ~(_2 | _848);
assign _850 = ~(_3 | _847);
assign _851 = ~(_0 | _846);
assign _852 = ~(_850 | _851);
assign _853 = ~(_1 | _852);
assign _856 = _849 | _853;
endmodule
