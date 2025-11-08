module m0x22 (input in2, in1, in3, output out);

	wire \$new_n5__0;

	not (\$new_n5__0, in2);
	nor (out, \$new_n5__0, in3);

endmodule
