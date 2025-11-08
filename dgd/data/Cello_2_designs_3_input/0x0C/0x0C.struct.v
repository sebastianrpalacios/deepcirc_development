module m0x0C (input in2, in1, in3, output out);

	wire \$n5_0;

	not (\$n5_0, in1);
	nor (out, in2, \$n5_0);

endmodule
