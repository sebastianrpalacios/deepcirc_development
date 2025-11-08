module gate(_0, _1, _2, _3, _191);
input _0, _1, _2, _3;
output _191;
wire _182, _183, _184, _185, _186, _187, _188, _189, _190;
assign _184 = ~_0;
assign _183 = ~_2;
assign _189 = ~(_0 | _2);
assign _182 = ~_3;
assign _186 = ~(_1 | _3);
assign _185 = ~(_182 | _184);
assign _187 = ~(_185 | _186);
assign _188 = ~(_183 | _187);
assign _190 = ~(_188 | _189);
assign _191 = _190;
endmodule
