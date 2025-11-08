module m0x13 (input in2, in1, in3, output out);

	wire \$new_n6__0;
	wire \$new_n5__0;

	not (\$new_n5__0, in2);
	nor (\$new_n6__0, in1, in3);
	nor (out, \$new_n6__0, \$new_n5__0);

endmodule
