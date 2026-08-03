import asyncio
import httpx
import uuid
import time
import random

API_BASE = "http://localhost:8000/api/v1"
CONCURRENT_USERS = 50

async def user_flow(user_idx: int) -> bool:
    unique_id = uuid.uuid4().hex[:8]
    email = f"loadtest_{user_idx}_{unique_id}@example.com"
    password = "password123"
    name = f"Load Test {user_idx}"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # 1. Register
            register_data = {
                "name": name,
                "email": email,
                "password": password
            }
            reg_res = await client.post(f"{API_BASE}/auth/register", data=register_data)
            if reg_res.status_code != 201:
                print(f"[{user_idx}] Registration failed: {reg_res.text}")
                return False
            
            token = reg_res.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            
            # 2. Create Session
            session_res = await client.post(f"{API_BASE}/sessions/", headers=headers)
            if session_res.status_code != 201:
                print(f"[{user_idx}] Session creation failed: {session_res.text}")
                return False
            
            session_id = session_res.json()["session_id"]
            
            # 3. Upload Document
            files = {
                "file": (f"test_doc_{user_idx}.txt", f"Hello world from {name}. This is a load test document.", "text/plain")
            }
            data = {
                "session_id": session_id
            }
            upload_res = await client.post(
                f"{API_BASE}/documents/upload", 
                headers=headers, 
                data=data, 
                files=files
            )
            if upload_res.status_code != 200:
                print(f"[{user_idx}] Document upload failed: {upload_res.text}")
                return False
                
            # 4. Delete User (cleanup)
            del_res = await client.delete(f"{API_BASE}/user/me", headers=headers)
            if del_res.status_code != 204:
                print(f"[{user_idx}] User deletion failed: {del_res.text}")
                return False
                
            print(f"[{user_idx}] Success flow complete")
            return True
            
        except Exception as e:
            print(f"[{user_idx}] Exception during flow: {str(e)}")
            return False

async def main():
    print(f"Starting load test with {CONCURRENT_USERS} concurrent users...")
    start_time = time.time()
    
    tasks = [user_flow(i) for i in range(CONCURRENT_USERS)]
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    duration = end_time - start_time
    
    success_count = sum(1 for r in results if r)
    fail_count = len(results) - success_count
    
    print("\nLoad Test Results:")
    print(f"Total time: {duration:.2f} seconds")
    print(f"Successful user flows: {success_count}/{CONCURRENT_USERS}")
    print(f"Failed user flows: {fail_count}/{CONCURRENT_USERS}")
    if duration > 0:
        print(f"Throughput: {CONCURRENT_USERS / duration:.2f} complete flows/second")

if __name__ == "__main__":
    asyncio.run(main())
