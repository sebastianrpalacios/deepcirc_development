module m0x4C (input in2, in1, in3, output out);

	wire \$n5_0;

	nor (\$n5_0, in1, in3);
	nor (out, in2, \$n5_0);

endmodule
