#include <iostream>
#include <vector>
using namespace std;

#define BLOCK_SIZE 256
#define QUANT_GROUP_SIZE 32
#define THREAD_GROUP_COUNT 4

#define THREAD_FLOAT_COUNT (THREAD_GROUP_COUNT * QUANT_GROUP_SIZE)
#define BLOCK_FLOAT_COUNT (BLOCK_SIZE * THREAD_FLOAT_COUNT)

int main()
{
    std::vector<int> visits(BLOCK_FLOAT_COUNT, 0);

    for (int i = 0; i < BLOCK_SIZE; i++) {
        int offset = (i * THREAD_FLOAT_COUNT) % BLOCK_FLOAT_COUNT;

        cout << "--- " << offset << ", i=" << i << ":" << endl;

        for (int j = 0; j < THREAD_FLOAT_COUNT; j++) {
            //int addr = (offset + j) % BLOCK_FLOAT_COUNT;
            int addr = offset + j;

            //cout << "    " << addr << endl;
            visits[addr]++;
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
