module m0x80 (input in2, in1, in3, output out);

	wire \$new_n6__0;
	wire \$new_n5__0;

	nor (\$new_n5__0, in2, in3);
	not (\$new_n6__0, \$new_n5__0);
	nor (out, \$new_n6__0, in1);

endmodule
