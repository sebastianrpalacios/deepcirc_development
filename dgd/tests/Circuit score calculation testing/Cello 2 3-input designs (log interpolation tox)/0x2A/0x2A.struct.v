module m0x2A (input in2, in1, in3, output out);

	wire \$new_n5__0;

	nor (\$new_n5__0, in2, in1);
	nor (out, \$new_n5__0, in3);

endmodule
