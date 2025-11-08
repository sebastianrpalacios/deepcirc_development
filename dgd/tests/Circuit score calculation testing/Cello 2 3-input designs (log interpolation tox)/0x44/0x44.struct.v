module m0x44 (input in2, in1, in3, output out);

	wire \$new_n5__0;

	not (\$new_n5__0, in3);
	nor (out, \$new_n5__0, in2);

endmodule
