module conv_buf #(parameter WIDTH = 28, DATA_BITS = 8)(
    input clk,
    input rst_n,
    input [DATA_BITS-1:0] data_in,
    input valid_in,
    output reg [DATA_BITS-1:0] w00, w01, w02,
    output reg [DATA_BITS-1:0] w10, w11, w12,
    output reg [DATA_BITS-1:0] w20, w21, w22,
    output reg valid_out
);
    reg [DATA_BITS-1:0] shift_reg [0 : 2*WIDTH + 2];
    reg [9:0] x_cnt, y_cnt;
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 0;
            x_cnt <= 0;
            y_cnt <= 0;
            for(i=0; i<=2*WIDTH+2; i=i+1) shift_reg[i] <= 0;
        end else if (valid_in) begin
            shift_reg[0] <= data_in;
            for(i=1; i<=2*WIDTH+2; i=i+1) shift_reg[i] <= shift_reg[i-1];
            
            w22 <= shift_reg[0]; w21 <= shift_reg[1]; w20 <= shift_reg[2];
            w12 <= shift_reg[WIDTH]; w11 <= shift_reg[WIDTH+1]; w10 <= shift_reg[WIDTH+2];
            w02 <= shift_reg[2*WIDTH]; w01 <= shift_reg[2*WIDTH+1]; w00 <= shift_reg[2*WIDTH+2];

            if (x_cnt == WIDTH-1) begin
                x_cnt <= 0;
                y_cnt <= y_cnt + 1;
            end else begin
                x_cnt <= x_cnt + 1;
            end
            
            if (y_cnt >= 2 && x_cnt >= 2) valid_out <= 1;
            else valid_out <= 0;
        end else begin
            valid_out <= 0;
        end
    end
endmodule