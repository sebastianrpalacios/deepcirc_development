module gate(_0, _1, _2, _3, _958);
input _0, _1, _2, _3;
output _958;
wire _949, _950, _951, _952, _953, _954, _955, _956, _957;
assign _950 = ~_1;
assign _951 = ~(_1 | _2);
assign _949 = ~_3;
assign _953 = ~(_0 | _950);
assign _952 = ~(_3 | _951);
assign _954 = ~(_2 | _949);
assign _955 = ~_954;
assign _956 = ~(_953 | _955);
assign _957 = ~(_952 | _956);
assign _958 = _957;
endmodule
