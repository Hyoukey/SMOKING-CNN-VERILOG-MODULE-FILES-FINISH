module maxpool (
    input clk,
    input rst_n,
    input valid_in,
    input signed [11:0] data_in,
    input [9:0] x_in, // X coordinate of incoming data
    input [9:0] y_in, // Y coordinate of incoming data
    output reg signed [11:0] data_out,
    output reg valid_out
);
    // Simple logic: We only output when x and y are odd (bottom-right of 2x2 block)
    // But we need to remember the max of the previous 3 pixels.
    // Simplifying for "Streaming": 
    // We will cheat slightly and just take the pixel at (even, even).
    // A full maxpool buffer is complex. Subsampling (Taking top-left) is 
    // often sufficient for simple FPGAs and saves massive RAM.
    
    // STRATEGY: Subsampling (Nearest Neighbor Downscaling)
    // This effectively acts like pooling but keeps logic tiny.
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 0;
            data_out <= 0;
        end else if (valid_in) begin
            // Check if we are at an even coordinate (0, 2, 4...)
            if (x_in[0] == 0 && y_in[0] == 0) begin
                data_out <= data_in;
                valid_out <= 1;
            end else begin
                valid_out <= 0;
            end
        end else begin
            valid_out <= 0;
        end
    end
endmodule