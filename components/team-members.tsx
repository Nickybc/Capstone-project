import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

export function TeamMembers() {
  return (
    <div className="grid gap-6 pt-8 md:grid-cols-2 lg:grid-cols-3">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle>Mentor</CardTitle>
          <CardDescription>Project Supervisor</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <Avatar className="h-16 w-16">
              <AvatarImage src="/placeholder.svg?height=64&width=64" alt="Zhang Jingrui" />
              <AvatarFallback>ZJ</AvatarFallback>
            </Avatar>
            <div>
              <h3 className="text-lg font-medium">Zhang Jingrui</h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">Professor</p>
            </div>
          </div>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle>Team Leader</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <Avatar className="h-16 w-16">
              <AvatarImage src="/placeholder.svg?height=64&width=64" alt="Yang Bochuang" />
              <AvatarFallback>YB</AvatarFallback>
            </Avatar>
            <div>
              <h3 className="text-lg font-medium">Yang Bochuang</h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">ID: 3036382143</p>
            </div>
          </div>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle>Team Member</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <Avatar className="h-16 w-16">
              <AvatarImage src="/placeholder.svg?height=64&width=64" alt="Peng Jiarui" />
              <AvatarFallback>PJ</AvatarFallback>
            </Avatar>
            <div>
              <h3 className="text-lg font-medium">Peng Jiarui</h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">ID: 3036381084</p>
            </div>
          </div>
        </CardContent>
      </Card>
      <Card className="md:col-span-2 lg:col-span-1">
        <CardHeader className="pb-2">
          <CardTitle>Team Member</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <Avatar className="h-16 w-16">
              <AvatarImage src="/placeholder.svg?height=64&width=64" alt="Ahebayan Yixian" />
              <AvatarFallback>AY</AvatarFallback>
            </Avatar>
            <div>
              <h3 className="text-lg font-medium">Ahebayan Yixian</h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">ID: 3036418605</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
