module m0x70 (input in2, in1, in3, output out);

	wire \$n5_0;

	nor (\$n5_0, in2, in3);
	nor (out, in1, \$n5_0);

endmodule
