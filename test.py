import asyncio

async def A():
    print('1')
    print('2')
    print('3')

async def B():
    print('4')
    print('5')
    print('6')

# loop = asyncio.get_event_loop()
# tasks = [A(), B()]
asyncio.run_coroutine_threadsafe(A())
asyncio.run(B())

# loop.close()