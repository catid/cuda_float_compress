#include <iostream>
#include <vector>
#include <cstdint>
using namespace std;

#define BLOCK_SIZE 256
#define QUANT_GROUP_SIZE 32
#define THREAD_GROUP_COUNT 4

#define THREAD_FLOAT_COUNT (THREAD_GROUP_COUNT * QUANT_GROUP_SIZE)
#define BLOCK_FLOAT_COUNT (BLOCK_SIZE * THREAD_FLOAT_COUNT)
#define INTERLEAVE_STRIDE (BLOCK_SIZE * THREAD_GROUP_COUNT)

int main()
{
    std::vector<int> visits(BLOCK_FLOAT_COUNT, 0);

    for (int i = 0; i < BLOCK_SIZE; i++) {

        cout << "--- i=" << i << ":" << endl;

        for (int j = 0; j < THREAD_GROUP_COUNT; j++) {
            for (int k = 0; k < QUANT_GROUP_SIZE; k++) {
                int addr = i * THREAD_GROUP_COUNT + j + k * INTERLEAVE_STRIDE;
                cout << "    " << addr << endl;
                visits[addr]++;
            }
        }
    }

    for (int i = 0; i < BLOCK_FLOAT_COUNT; i++) {
        if (visits[i] != 1) {
            cout << "ERROR: offset=" << i << " count = " << visits[i] << endl;
            return 1;
        }
    }

    cout << "All addresses visited exactly once." << endl;

    return 0;
}
